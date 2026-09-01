"""What does the terminal value of stored energy do to the plan?

The optimiser scores whatever is left in the battery at the end of its horizon at
one price, p_a. EOS_connect passes the forensic cost of the stored energy - what
it happened to cost when it was bought. That is a sunk cost. The economically
right figure is what the energy would cost to *replace* after the horizon, and
whenever p_a sits below the cheapest price in the horizon, keeping a kWh is worth
less than buying one, so discharging wins in every slot and the plan drains to
the floor regardless of what follows.

This re-solves real requests with different terminal values and scores each plan
on one yardstick, because objective values are not comparable across p_a: raising
it raises the objective by construction. The yardstick is

    grid cost inside the horizon  -  (value of the final stock - value of the
    stock it started with)

where a stock is worth the replacement price up to what the house will draw after
the horizon, and only the feed-in tariff beyond that.

Three ways to get that yardstick wrong, all of them tried first: crediting
leftover energy at the replacement price without limit makes hoarding pay without
bound, so the best p_a is simply the largest. Crediting the final stock outright
credits the stock the plan started with, which it never bought. And the load
series is already energy per slot, so multiplying by the slot length again
understates the consumption that can use the leftover energy by a factor of four.

One request is a snapshot, and its answer is dominated by that day's spread
between the cheapest price inside the horizon and the price after it. Run it over
the archive instead: solves are cached per request, so a daily run only pays for
the days that are new.

Read-only. Solves copies; changes nothing in the running instance.
"""

import argparse
import copy
import glob
import gzip
import json
import os
import re
import sys
from datetime import datetime

# The tool lives beside the package it re-solves with, so the image needs no
# separate copy step and the version always matches the running one.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces.optimization_backends.optimization_backend_local_evopt import (  # noqa: E402
    LocalEVOptBackend,
)

LIVE_REQUEST = "/app/json/optimize_request_local_evopt.json"
ARCHIVE_DEFAULT = "/archive"
NAME_RE = re.compile(r"^(\d{8}T\d{6})-optimize_request_local_evopt\.json\.gz$")
LABELS = ["Einstand (aktuell)", "Wiederbeschaffung", "Wiederbeschaffung +20%"]


def solve(request, p_a):
    """Solve a copy of the request with the terminal value replaced."""
    work = copy.deepcopy(request)
    for battery in work.get("batteries", []):
        battery["p_a"] = p_a
    backend = object.__new__(LocalEVOptBackend)
    backend.time_frame_base = 900
    backend.num_threads = None
    backend.time_limit = None
    backend.charging_strategy = "charge_before_export"
    backend.discharging_strategy = "discharge_before_import"
    backend.emergency_reserve_pct = 0
    backend.max_grid_import_w = None
    backend.max_grid_export_w = None
    from interfaces.optimization_backends.local_evopt.optimizer import probe_cbc_solver

    backend.cbc_path = probe_cbc_solver()
    return backend._build_optimizer(work, 180).solve()


def evaluate(request):
    """Score the plans a set of terminal values produces for one request."""
    ts = request["time_series"]
    prices, durations, loads = ts["p_N"], ts["dt"], ts["gt"]
    battery = request["batteries"][0]
    capacity, s_min = battery["s_max"], battery["s_min"]
    s_initial, current = battery["s_initial"], battery["p_a"]

    # Cheapest the energy could be bought for just past the horizon: what a kWh
    # left in the battery actually saves.
    tail = prices[-48:] if len(prices) > 48 else prices
    replacement = sum(sorted(tail)[:12]) / 12

    # gt is already energy per slot. A day is the honest window: the battery is
    # sized for roughly that, and what it still holds afterwards can only be sold.
    slot_hours = durations[-1] / 3600.0
    post_horizon_kwh = (sum(loads[-48:]) / 48) * (24.0 / slot_hours) / 1000.0
    eta_d = float(request.get("eta_d", 0.9))
    eta_c = float(request.get("eta_c", 0.9))
    feed_in = min(ts.get("p_E") or [0.0])

    def stock_value(stock_wh):
        usable = max(0.0, stock_wh - s_min) / 1000.0 * eta_d
        displaces = min(usable, post_horizon_kwh)
        surplus = max(0.0, usable - post_horizon_kwh)
        return (displaces * replacement + surplus * feed_in) * 1000.0

    out = {
        "replacement_ct": replacement * 1e5,
        "cheapest_ct": min(prices) * 1e5,
        # Buying at the cheapest price and getting it back through the round trip.
        # Above the replacement price, holding cannot pay whatever p_a says - this
        # is the one comparison that needs no assumption about the yardstick.
        "effective_ct": min(prices) / (eta_c * eta_d) * 1e5,
        "variants": {},
    }
    for label, p_a in zip(LABELS, [current, replacement, replacement * 1.2]):
        result = solve(request, p_a)
        soc = result["batteries"][0]["state_of_charge"]
        imports = result["grid_import"]
        grid_cost = sum(
            v * d / 3600.0 * p for v, d, p in zip(imports, durations, prices)
        )
        out["variants"][label] = {
            "p_a_ct": p_a * 1e5,
            "soc_end_pct": soc[-1] / capacity * 100,
            "soc_min_pct": min(soc) / capacity * 100,
            "grid_kwh": sum(v * d / 3600.0 for v, d in zip(imports, durations)) / 1000.0,
            "score_eur": grid_cost - (stock_value(soc[-1]) - stock_value(s_initial)),
        }
    return out


def pick_daily(paths):
    """One request per day, the one closest to midday: the horizon is then long
    enough to matter and still mostly backed by settled prices."""
    by_day = {}
    for path in paths:
        match = NAME_RE.match(os.path.basename(path))
        if not match:
            continue
        stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        key = stamp.date().isoformat()
        distance = abs(stamp.hour * 60 + stamp.minute - 12 * 60)
        if key not in by_day or distance < by_day[key][0]:
            by_day[key] = (distance, path)
    return [(day, path) for day, (_, path) in sorted(by_day.items())]


def main():
    parser = argparse.ArgumentParser(description="Terminal value sweep.")
    parser.add_argument("--archive", default=ARCHIVE_DEFAULT)
    parser.add_argument("--live", action="store_true", help="only the current request")
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    cache = {}
    if args.cache and os.path.exists(args.cache):
        try:
            with open(args.cache, encoding="utf-8") as handle:
                cache = json.load(handle)
        except (OSError, ValueError):
            cache = {}   # a damaged cache costs solver time, not correctness

    days = []
    if args.live:
        with open(LIVE_REQUEST, encoding="utf-8") as handle:
            days.append(("jetzt", evaluate(json.load(handle))))
    else:
        selected = pick_daily(sorted(glob.glob(os.path.join(args.archive, "*.json.gz"))))
        if not selected:
            print(f"Keine archivierten Anfragen unter {args.archive}.")
            return
        for day, path in selected:
            key = os.path.basename(path)
            if key in cache:
                days.append((day, cache[key]))
                continue
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    result = evaluate(json.load(handle))
            except (OSError, ValueError, KeyError, IndexError) as exc:
                print(f"  {day}: uebersprungen ({exc})")
                continue
            cache[key] = result
            days.append((day, result))

        if args.cache:
            try:
                with open(args.cache, "w", encoding="utf-8") as handle:
                    json.dump(cache, handle)
            except OSError as exc:
                print(f"Cache nicht schreibbar: {exc}")

    print("%-12s %9s %9s %9s   %s" % (
        "Tag", "billigst", "effektiv", "Ersatz",
        "  ".join("%22s" % label[:22] for label in LABELS)))
    print("%-12s %9s %9s %9s   %s" % (
        "", "ct/kWh", "ct/kWh", "ct/kWh",
        "  ".join("%22s" % "SoC Ende / Bewertung" for _ in LABELS)))

    totals = {label: 0.0 for label in LABELS}
    wins = {label: 0 for label in LABELS}
    for day, result in days:
        cells, best, best_label = [], None, None
        for label in LABELS:
            v = result["variants"][label]
            cells.append("%13.1f %% %8.3f" % (v["soc_end_pct"], v["score_eur"]))
            totals[label] += v["score_eur"]
            if best is None or v["score_eur"] < best:
                best, best_label = v["score_eur"], label
        wins[best_label] += 1
        print("%-12s %9.1f %9.1f %9.1f   %s" % (
            day, result["cheapest_ct"], result["effective_ct"],
            result["replacement_ct"], "  ".join(cells)))

    if len(days) > 1:
        print()
        print("%-28s %12s %14s" % ("Variante", "Summe EUR", "beste an"))
        for label in LABELS:
            print("%-28s %12.3f %8d Tagen" % (label, totals[label], wins[label]))
        pays = sum(1 for _, r in days if r["effective_ct"] < r["replacement_ct"])
        print()
        print("An %d von %d Tagen lag der effektive Speicherpreis unter dem Ersatzpreis"
              % (pays, len(days)))
        print("danach - nur dort kann Halten ueberhaupt zahlen, unabhaengig vom Massstab.")


if __name__ == "__main__":
    main()
