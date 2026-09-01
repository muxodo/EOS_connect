"""MILP against genetic algorithm, on the same real requests.

EOS_connect started out talking to Akkudoktor EOS and later switched to the
vendored evcc MILP optimiser. The stated reasons were operational - in-process,
fast, no second service - not that the plans were better. So the question of
which method plans better for this house was never asked, and it matters:
battery degradation, a non-linear COP curve and the value of energy beyond the
horizon are all things a linear model cannot express and a simulating optimiser
can.

What is compared
----------------
Three variants of the same archived request:

    MILP 15 min   what runs in production
    MILP hourly   the same solver on hourly data
    GA hourly     Akkudoktor EOS

The middle one exists so the comparison is not confounded: EOS only plans
hourly, so any difference against the 15-minute MILP is partly resolution and
partly method. MILP-hourly against GA-hourly isolates the method.

How they are scored
-------------------
Not by their own accounting. This EOS build leaves grid consumption and costs at
zero in its response while returning a real state-of-charge trajectory, and the
two response formats disagree anyway. Only the *decision* is taken from each -
the SoC path - and the energy balance is then simulated identically for both:

    battery flow -> AC, via the charge and discharge efficiencies
    grid         = load - pv + what the battery takes - what it gives
    cost         = imported energy * the price of that slot

The yardstick is the one from measure_p_a: cost inside the horizon, less the
change in the value of what is stored, where stored energy is worth the
replacement price up to what the house will draw afterwards and only the feed-in
tariff beyond that.

Runs inside the eos-connect container: it needs the vendored optimiser and its
solver, and reaches Akkudoktor EOS over the network.
"""

import argparse
import copy
import glob
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytz  # noqa: E402

from interfaces.optimization_backends.optimization_backend_local_evopt import (  # noqa: E402
    LocalEVOptBackend,
)

ARCHIVE_DEFAULT = "/archive"
EOS_URL_DEFAULT = "http://192.168.0.201:8503/optimize"
NAME_RE = re.compile(r"^(\d{8}T\d{6})-optimize_request\.json\.gz$")


def to_hourly(request):
    """Aggregate the quarter-hour arrays to hours, as EOS only plans hourly."""
    work = copy.deepcopy(request)
    ems = work["ems"]

    def fold(values, energy):
        out = []
        for hour in range(len(values) // 4):
            block = values[hour * 4:hour * 4 + 4]
            out.append(sum(block) if energy else sum(block) / len(block))
        return out

    for key in ("pv_prognose_wh", "gesamtlast"):
        ems[key] = fold(ems[key], True)
    for key in ("strompreis_euro_pro_wh", "einspeiseverguetung_euro_pro_wh"):
        ems[key] = fold(ems[key], False)
    if work.get("temperature_forecast"):
        work["temperature_forecast"] = fold(work["temperature_forecast"], False)
    return work


def for_eos(hourly_request):
    """Rename and complete the payload for the current EOS API.

    EOS_connect's payload predates several renames: devices are pv_battery and ev
    rather than pv_akku and eauto, each needs a device_id, and the inverter has to
    name the battery it belongs to.
    """
    work = copy.deepcopy(hourly_request)
    work["pv_battery"] = dict(work.pop("pv_akku", None) or {})
    work["pv_battery"].setdefault("device_id", "battery1")
    work["ev"] = dict(work.pop("eauto", None) or {})
    work["ev"].setdefault("device_id", "ev1")
    work["inverter"] = dict(work.get("inverter") or {})
    work["inverter"].setdefault("device_id", "inverter1")
    work["inverter"]["battery_id"] = work["pv_battery"]["device_id"]
    if work.get("dishwasher"):
        work["dishwasher"] = dict(work["dishwasher"])
        work["dishwasher"].setdefault("device_id", "dishwasher1")
    work.pop("start_solution", None)
    return work


def run_milp(request, time_frame_base):
    backend = LocalEVOptBackend(
        time_frame_base=time_frame_base,
        time_zone=pytz.timezone(os.environ.get("TZ_NAME", "Europe/Berlin")),
    )
    started = time.time()
    response, _ = backend.optimize(copy.deepcopy(request))
    return response.get("result", {}).get("akku_soc_pro_stunde") or [], time.time() - started


def run_genetic(request, url):
    payload = json.dumps(for_eos(request)).encode("utf-8")
    call = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    started = time.time()
    with urllib.request.urlopen(call, timeout=600) as answer:
        body = json.load(answer)
    return body.get("result", {}).get("akku_soc_pro_stunde") or [], time.time() - started


def simulate(soc_pct, pv, load, prices, capacity_wh, eta_c, eta_d, slot_hours):
    """Energy balance of a plan, from its state of charge alone.

    Deliberately independent of what either optimiser reports about itself: this
    EOS build returns zeros for grid consumption and costs, and the two response
    formats do not agree. The trajectory is the decision; everything else follows
    from it and from the inputs both were given.
    """
    n = min(len(soc_pct), len(pv), len(load), len(prices))
    if n < 2:
        return None
    imported_kwh = 0.0
    cost = 0.0
    for i in range(1, n):
        stored_delta = (soc_pct[i] - soc_pct[i - 1]) / 100.0 * capacity_wh
        if stored_delta >= 0:
            # Charging: more has to arrive at the AC side than lands in the cell.
            battery_ac = stored_delta / eta_c
        else:
            battery_ac = stored_delta * eta_d
        grid = load[i] - pv[i] + battery_ac
        if grid > 0:
            imported_kwh += grid / 1000.0
            cost += grid / 1000.0 * prices[i] * 1000.0
    return imported_kwh, cost, soc_pct[0], soc_pct[n - 1]


def score_plan(result, request_hourly, replacement, post_horizon_kwh, feed_in,
               capacity_wh, eta_d, s_min_pct):
    imported_kwh, cost, soc_start, soc_end = result

    def stock_value(soc_percent):
        usable = max(0.0, soc_percent - s_min_pct) / 100.0 * capacity_wh / 1000.0 * eta_d
        displaces = min(usable, post_horizon_kwh)
        surplus = max(0.0, usable - post_horizon_kwh)
        return displaces * replacement * 1000.0 + surplus * feed_in * 1000.0

    return cost - (stock_value(soc_end) - stock_value(soc_start))


def compare(request, eos_url):
    hourly = to_hourly(request)
    ems = hourly["ems"]
    prices, pv, load = (
        ems["strompreis_euro_pro_wh"], ems["pv_prognose_wh"], ems["gesamtlast"]
    )
    battery = request.get("pv_akku") or {}
    capacity_wh = float(battery.get("capacity_wh") or 0)
    eta_c = float(battery.get("charging_efficiency") or 0.9)
    eta_d = float(battery.get("discharging_efficiency") or 0.9)
    s_min_pct = float(battery.get("min_soc_percentage") or 0)

    replacement = sum(sorted(prices[-12:])[:6]) / 6
    post_horizon_kwh = (sum(load[-12:]) / 12) * 24 / 1000.0
    feed_in = min(ems["einspeiseverguetung_euro_pro_wh"])

    variants = {}
    for label, runner in (
        ("MILP 15 min", lambda: run_milp(request, 900)),
        ("MILP stuendlich", lambda: run_milp(hourly, 3600)),
        ("GA stuendlich", lambda: run_genetic(hourly, eos_url)),
    ):
        try:
            soc, seconds = runner()
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            variants[label] = {"error": str(exc)[:70]}
            continue
        if not soc:
            variants[label] = {"error": "keine SoC-Reihe"}
            continue
        # The 15-minute plan is folded to hours so all three are simulated on the
        # same series; otherwise the comparison would measure the grid, not the plan.
        series = soc[::4] if label.startswith("MILP 15") else soc
        outcome = simulate(series, pv, load, prices, capacity_wh, eta_c, eta_d, 1.0)
        if outcome is None:
            variants[label] = {"error": "zu kurz"}
            continue
        variants[label] = {
            "seconds": seconds,
            "grid_kwh": outcome[0],
            "soc_end": outcome[3],
            "score": score_plan(outcome, hourly, replacement, post_horizon_kwh,
                                feed_in, capacity_wh, eta_d, s_min_pct),
        }
    return {
        "replacement_ct": replacement * 1e5,
        "cheapest_ct": min(prices) * 1e5,
        "variants": variants,
    }


def main():
    parser = argparse.ArgumentParser(description="MILP against genetic algorithm.")
    parser.add_argument("--archive", default=ARCHIVE_DEFAULT)
    parser.add_argument("--eos-url", default=EOS_URL_DEFAULT)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--live", default=None, help="path of a single request")
    args = parser.parse_args()

    cache = {}
    if args.cache and os.path.exists(args.cache):
        try:
            with open(args.cache, encoding="utf-8") as handle:
                cache = json.load(handle)
        except (OSError, ValueError):
            cache = {}

    jobs = []
    if args.live:
        with open(args.live, encoding="utf-8") as handle:
            jobs.append(("jetzt", json.load(handle), None))
    else:
        by_day = {}
        for path in sorted(glob.glob(os.path.join(args.archive, "*.json.gz"))):
            match = NAME_RE.match(os.path.basename(path))
            if not match:
                continue
            stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
            distance = abs(stamp.hour * 60 + stamp.minute - 12 * 60)
            key = stamp.date().isoformat()
            if key not in by_day or distance < by_day[key][0]:
                by_day[key] = (distance, path)
        for day, (_, path) in sorted(by_day.items()):
            jobs.append((day, None, path))

    labels = ["MILP 15 min", "MILP stuendlich", "GA stuendlich"]
    print("%-12s %9s %9s   %s" % (
        "Tag", "billigst", "Ersatz", "  ".join("%26s" % l for l in labels)))
    print("%-12s %9s %9s   %s" % (
        "", "ct/kWh", "ct/kWh",
        "  ".join("%26s" % "Netz kWh / EUR / Sek" for _ in labels)))

    rows = []
    for day, inline, path in jobs:
        key = os.path.basename(path) if path else None
        if key and key in cache:
            result = cache[key]
        else:
            try:
                if inline is None:
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        inline = json.load(handle)
                result = compare(inline, args.eos_url)
            except (OSError, ValueError, KeyError) as exc:
                print(f"  {day}: uebersprungen ({exc})")
                continue
            if key:
                cache[key] = result
        rows.append((day, result))
        cells = []
        for label in labels:
            v = result["variants"].get(label, {})
            if "error" in v:
                cells.append("%26s" % v["error"][:26])
            else:
                cells.append("%9.2f %8.3f %6.1f" % (v["grid_kwh"], v["score"], v["seconds"]))
        print("%-12s %9.1f %9.1f   %s" % (
            day, result["cheapest_ct"], result["replacement_ct"], "  ".join(cells)))

    if args.cache:
        try:
            with open(args.cache, "w", encoding="utf-8") as handle:
                json.dump(cache, handle)
        except OSError as exc:
            print(f"Cache nicht schreibbar: {exc}")

    usable = [r for _, r in rows
              if all("error" not in r["variants"].get(l, {"error": 1}) for l in labels)]
    if len(usable) > 1:
        print()
        print("%-20s %12s %10s" % ("Variante", "Summe EUR", "beste an"))
        for label in labels:
            total = sum(r["variants"][label]["score"] for r in usable)
            wins = sum(1 for r in usable
                       if min(r["variants"][l]["score"] for l in labels)
                       == r["variants"][label]["score"])
            print("%-20s %12.3f %6d Tagen" % (label, total, wins))
        print()
        print("MILP stuendlich gegen GA stuendlich ist der Methodenvergleich.")
        print("Der Abstand zu MILP 15 min zeigt, was allein die Aufloesung bringt.")


if __name__ == "__main__":
    main()
