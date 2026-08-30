"""Local verification of the temperature-aware heat pump load correction.

Runs entirely offline: the regression and the profile swap are pure arithmetic,
so they can be checked without Home Assistant or a live instance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from interfaces.heatpump_model import (  # noqa: E402
    apply_correction,
    fit_heating_regression,
    predict_daily_w,
)

SLOTS = 96  # one day at 15-minute resolution


def synth_samples(per_degree=46.0, base_t=15.0, intercept=0.0):
    """Daily (temperature, heat pump power) pairs from a known relationship, so
    the fit can be checked against the truth that produced them.

    ``per_degree`` is watts per *heating degree*, i.e. positive: the colder it
    gets below base_t, the more the heat pump draws.
    """
    out = []
    for temp in (-5, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22):
        out.append((float(temp), intercept + per_degree * max(0.0, base_t - temp)))
    return out


def main() -> int:
    fails = []

    # 1. The fit must recover a known slope and heating limit.
    fit = fit_heating_regression(synth_samples())
    ok = fit is not None and abs(fit[1] - 46.0) < 1.0 and abs(fit[2] - 15.0) < 2.1
    print(f"1. Regression findet Steigung/Heizgrenze: {'OK' if ok else 'FEHLER'}  "
          f"{'' if fit is None else f'{fit[1]:+.1f} W je Heizgrad unter {fit[2]:.0f} C'}")
    if not ok:
        fails.append("fit")

    # 2. Colder must predict more, and above the heating limit it must be zero.
    if fit:
        cold, mild, warm = predict_daily_w(fit, -5), predict_daily_w(fit, 8), predict_daily_w(fit, 25)
        ok = cold > mild > 0 and warm == 0.0
        print(f"2. Vorhersage monoton, ueber Heizgrenze null: {'OK' if ok else 'FEHLER'}  "
              f"(-5C {cold:.0f} W, 8C {mild:.0f} W, 25C {warm:.0f} W)")
        if not ok:
            fails.append("monotonic")

    # 3. Refuse a fit that says the heat pump runs harder when it is warm - that
    # is not heating, and trusting it would push the forecast the wrong way.
    inverted = [(t, 10.0 * t) for t in range(0, 20)]
    ok = fit_heating_regression(inverted) is None
    print(f"3. Verwirft positiven Zusammenhang: {'OK' if ok else 'FEHLER'}")
    if not ok:
        fails.append("sign guard")

    # 4. Too little data must yield no fit rather than a fragile one.
    ok = fit_heating_regression([(5.0, 500.0), (6.0, 450.0)]) is None
    print(f"4. Zu wenige Tage -> kein Fit: {'OK' if ok else 'FEHLER'}")
    if not ok:
        fails.append("min samples")

    # 5. The correction must keep the reference shape and hit the predicted mean.
    profile = [100.0] * SLOTS
    reference = [50.0 if 0 <= i < SLOTS // 2 else 150.0 for i in range(SLOTS)]  # mean 100
    corrected = apply_correction(profile, reference, [200.0], SLOTS)
    new_hp = [c - p + r for c, p, r in zip(corrected, profile, reference)]
    mean_ok = abs(sum(new_hp) / len(new_hp) - 200.0) < 1.0
    shape_ok = abs((new_hp[0] / new_hp[-1]) - (reference[0] / reference[-1])) < 1e-6
    print(f"5. Austausch trifft Zielmittel und haelt die Form: "
          f"{'OK' if mean_ok and shape_ok else 'FEHLER'} (Mittel {sum(new_hp)/len(new_hp):.0f} W)")
    if not (mean_ok and shape_ok):
        fails.append("correction")

    # 6. Never negative, even when the model wants to remove more than is there.
    corrected = apply_correction([10.0] * SLOTS, [500.0] * SLOTS, [0.0], SLOTS)
    ok = all(v >= 0 for v in corrected)
    print(f"6. Bleibt nicht-negativ: {'OK' if ok else 'FEHLER'} (min {min(corrected):.1f})")
    if not ok:
        fails.append("negative")

    # 7. A reference week with no heating has no shape worth scaling, so the
    # predicted amount is spread evenly instead of amplifying noise.
    corrected = apply_correction([100.0] * SLOTS, [1.0] * SLOTS, [300.0], SLOTS)
    new_hp = [c - 100.0 + 1.0 for c in corrected]
    ok = all(abs(v - 300.0) < 1.0 for v in new_hp)
    print(f"7. Ohne Heizbetrieb in der Referenz gleichmaessig verteilt: {'OK' if ok else 'FEHLER'}")
    if not ok:
        fails.append("flat fallback")

    # 8. Missing prerequisites must return the profile untouched, not a guess.
    same = apply_correction(profile, [], [200.0], SLOTS)
    ok = same == profile and apply_correction(profile, reference, [None], SLOTS) == profile
    print(f"8. Fehlende Daten -> Profil unveraendert: {'OK' if ok else 'FEHLER'}")
    if not ok:
        fails.append("passthrough")

    print()
    print("ERGEBNIS:", "alle Pruefungen bestanden" if not fails else f"FEHLGESCHLAGEN: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
