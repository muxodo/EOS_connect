"""
Module: heatpump_model

Temperature-aware correction of the household load forecast for an unschedulable
heat pump.

The load profile in load_interface.py is built purely from history (the same
weekday one and two weeks back). A heat pump is a large, weather-driven part of
that load and cannot be planned away, so a cold snap tomorrow is invisible to a
forecast that only looks backwards - which is exactly when the schedule matters.

Measured on one installation over 129 days of heating season: the heat pump
correlates -0.845 with outdoor temperature at -46 W/K and makes up 32% of the
winter load. Regressing the *total* load on temperature barely helps (+1.7%),
because the temperature-independent household share drowns the signal. Modelling
the heat pump separately and leaving the rest to history cut the daily forecast
error from 299 W to 191 W (-36%), better on 71% of days.

The correction replaces only the heat pump's share and leaves the historical
profile's shape intact:

    forecast[t] = load_profile[t] - hp_reference[t] + hp_predicted[t]
"""

import logging

logger = logging.getLogger("__main__")

# Candidate heating base temperatures. The right one is picked by fit quality
# rather than assumed, because it depends on the building and the heating curve;
# on the reference installation the choice landed between 10 and 15 degC.
BASE_TEMPERATURE_CANDIDATES = (8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 18.0)

# Below this daily average the reference days had essentially no heating, so
# scaling their shape would amplify noise instead of carrying information.
MIN_REFERENCE_W = 20.0

# Bounds on the scale applied to the reference shape. A mild reference week
# against a hard frost would otherwise produce an arbitrarily large factor.
MIN_SCALE, MAX_SCALE = 0.0, 5.0

# A slope is only meaningful if the training window actually contains heating.
# Fitted on days that all sat near or above the heating limit, the slope is noise,
# and the prediction then extrapolates that noise down to winter temperatures the
# data never covered. Require a few genuinely cold days and some spread between
# them before the fit is trusted.
MIN_HEATING_DAYS = 3
MIN_HEATING_DEGREES = 2.0
MIN_DEGREE_SPREAD = 4.0


def heating_degrees(temperatures, base_t):
    """Mean heating degrees over a day, integrated across the temperature curve.

    Deliberately not ``max(0, base - mean(temperatures))``. Heating degrees are a
    kinked function, so averaging the temperature first and clipping afterwards is
    not the same as clipping each slot and averaging: warm hours cancel cold ones
    instead of being cut off at zero. A day at 14 degC by noon and 4 degC at night
    reads as 3 degrees below a 12 degC limit on the daily mean, but noticeably more
    when each slot is counted, and that difference is largest in the shoulder
    seasons where the temperature crosses the heating limit daily.

    Measured over 120 winter days here, integrating cut the day-ahead error from
    77.3 W to 70.5 W (-8.8%). Adding the daily minimum as a second regressor made
    it worse, at 78.1 W - the information is already in here.

    Returns None if the day has no usable readings.
    """
    usable = [t for t in temperatures if t is not None]
    if not usable:
        return None
    return sum(max(0.0, base_t - t) for t in usable) / len(usable)


def fit_heating_regression(samples):
    """Least-squares fit of heat pump power against heating degrees.

    ``samples`` is a sequence of (temperatures, heatpump_w) per day, where
    ``temperatures`` is that day's outdoor temperature curve (one entry per slot)
    and ``heatpump_w`` its average power. Returns ``(intercept, slope,
    base_temperature)`` or None if the data cannot support a fit.

    Heating degrees rather than raw temperature: above the heating limit the heat
    pump stops responding to temperature at all, and a single straight line
    through both regimes fits neither of them.
    """
    usable = [
        (temps, w)
        for temps, w in samples
        if temps is not None and w is not None and any(t is not None for t in temps)
    ]
    if len(usable) < 5:
        logger.debug("[HP-MODEL] Only %d usable samples - no fit", len(usable))
        return None

    best = None
    for base_t in BASE_TEMPERATURE_CANDIDATES:
        pairs = [
            (heating_degrees(temps, base_t), w)
            for temps, w in usable
        ]
        pairs = [(x, y) for x, y in pairs if x is not None]
        if len(pairs) < 5:
            continue
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom <= 0:
            # Every day sat on the same side of this base temperature, so the
            # slope is unidentifiable here - a different candidate may still work.
            continue
        heating_days = [x for x in xs if x >= MIN_HEATING_DEGREES]
        if len(heating_days) < MIN_HEATING_DAYS or (max(xs) - min(xs)) < MIN_DEGREE_SPREAD:
            # Too little heating in the window to tell a slope from noise.
            continue
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        intercept = mean_y - slope * mean_x
        sse = sum((intercept + slope * x - y) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[0]:
            best = (sse, intercept, slope, base_t)

    if best is None:
        logger.info(
            "[HP-MODEL] No usable fit: the %d-day window contains too little heating "
            "to establish a temperature response",
            len(usable),
        )
        return None
    _, intercept, slope, base_t = best
    if slope <= 0:
        # The slope is per *heating degree*, so heating means a positive slope:
        # more degrees below the limit, more consumption. A negative one says the
        # heat pump works harder as it warms up, which is not heating - trusting
        # it would push the forecast the wrong way on exactly the cold days the
        # model exists for.
        logger.info(
            "[HP-MODEL] Discarding fit: consumption falls with colder weather (%.1f W/K)",
            slope,
        )
        return None
    return intercept, slope, base_t


def predict_daily_w(fit, temperatures):
    """Predicted daily average heat pump power for a day's temperature curve.

    ``temperatures`` is the forecast for that day at the profile's resolution, so
    the heating degrees are integrated the same way the fit was trained.
    """
    if fit is None or not temperatures:
        return None
    intercept, slope, base_t = fit
    degrees = heating_degrees(temperatures, base_t)
    if degrees is None:
        return None
    return max(0.0, intercept + slope * degrees)


def apply_correction(
    load_profile, hp_reference, predicted_daily_w, slots_per_day, slot_hours
):
    """Swap the heat pump's historical share for the temperature-based one.

    ``load_profile`` and ``hp_reference`` are per-slot lists of *energy* in the
    same unit and of the same length; ``predicted_daily_w`` holds one predicted
    daily average *power* per day covered by the profile (typically today and
    tomorrow); ``slot_hours`` is the length of one slot in hours, which is what
    converts between the two.

    That conversion is the whole reason this argument exists. Without it a 500 W
    prediction was written into a profile counting watt-hours per quarter hour,
    inflating the forecast fourfold.

    The reference *shape* is kept and only rescaled, because heating is not spread
    evenly over the day - rebuilding it from a daily average alone would discard
    that pattern. Where the reference days had no meaningful heating there is no
    shape worth keeping, so the predicted amount is spread evenly instead.
    """
    if not load_profile or not hp_reference or len(hp_reference) != len(load_profile):
        return load_profile
    if not slot_hours or slot_hours <= 0:
        return load_profile

    corrected = list(load_profile)
    for day_index, predicted_w in enumerate(predicted_daily_w):
        start = day_index * slots_per_day
        end = min(start + slots_per_day, len(load_profile))
        if start >= len(load_profile) or predicted_w is None:
            continue

        window = hp_reference[start:end]
        if not window:
            continue

        # Everything below is per slot, in the profile's energy unit.
        predicted_per_slot = predicted_w * slot_hours
        reference_per_slot = sum(window) / len(window)
        reference_power_w = reference_per_slot / slot_hours

        if reference_power_w > MIN_REFERENCE_W:
            scale = max(MIN_SCALE, min(MAX_SCALE, predicted_per_slot / reference_per_slot))
            new_window = [v * scale for v in window]
        else:
            new_window = [predicted_per_slot] * len(window)

        for offset, (old, new) in enumerate(zip(window, new_window)):
            # Never below zero: the household still consumes something even if the
            # model wanted to subtract more heat pump than the profile contains.
            corrected[start + offset] = max(0.0, corrected[start + offset] - old + new)

    return corrected
