"""What does the terminal value of stored energy do to the plan?

The optimiser scores whatever is left in the battery at the end of its horizon at
one price, p_a. EOS_connect passes the forensic cost of the stored energy - what
it happened to cost when it was bought. That is a sunk cost. The economically
right figure is what the energy would cost to *replace* after the horizon, and
when p_a sits below the cheapest price in the horizon, keeping a kWh is worth
less than buying one, so the plan drains to the floor in every slot.

This sweeps p_a over the live request and scores each resulting plan on a common
yardstick, because comparing objective values across different p_a is
meaningless: raising p_a raises the objective by construction. The yardstick is

    grid cost inside the horizon  -  (value of the final stock - value of the
    stock it started with)

where a stock is worth the replacement price up to what the house will draw
after the horizon, and only the feed-in tariff beyond that.

The cap matters. Valuing every leftover kWh at the replacement price makes
hoarding pay without limit, so the "best" p_a would simply be the largest one -
the yardstick would be measuring its own assumption. Energy is only worth a
displaced purchase up to the amount the house will draw before the next cheap
window; past that it can only be exported. Round-trip losses are charged too,
since only part of what is stored comes back out.

Crediting the final stock outright would be just as wrong: the battery does not
start empty, so most of what is left at the end was never bought by this plan.
Only the change in stock belongs to it.

Read-only; solves copies of the request that was already built, and changes
nothing in the running instance.
"""

import copy
import json
import sys

sys.path.insert(0, "/app")

from interfaces.optimization_backends.optimization_backend_local_evopt import (  # noqa: E402
    LocalEVOptBackend,
)

REQUEST = "/app/json/optimize_request_local_evopt.json"


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


def main():
    with open(REQUEST, encoding="utf-8") as handle:
        request = json.load(handle)

    prices = request["time_series"]["p_N"]          # EUR/Wh
    durations = request["time_series"]["dt"]
    capacity = request["batteries"][0]["s_max"]
    s_min = request["batteries"][0]["s_min"]
    current = request["batteries"][0]["p_a"]

    # The replacement price: the cheapest the energy could be bought for in the
    # stretch just past the horizon. That is what leftover energy actually saves.
    tail = prices[-48:] if len(prices) > 48 else prices
    replacement = sum(sorted(tail)[:12]) / 12

    # How much the house will draw after the horizon before it could refill
    # cheaply. Only that much leftover energy can displace a purchase.
    # gt is already energy per slot, so the mean slot times the number of slots
    # in a day gives the kWh the house will draw before it could plausibly refill
    # cheaply again. A day is the honest window: the battery is sized for roughly
    # that, and anything it still holds after it could only be exported.
    loads = request["time_series"]["gt"]
    slot_hours = durations[-1] / 3600.0
    mean_slot_wh = sum(loads[-48:]) / 48
    post_horizon_kwh = mean_slot_wh * (24.0 / slot_hours) / 1000.0
    eta_d = float(request.get("eta_d", 0.9))
    feed_in = min(request["time_series"].get("p_E") or [0.0])
    s_initial = request["batteries"][0]["s_initial"]

    def stock_value(stock_wh):
        """What a given state of charge is worth once the horizon ends."""
        usable = max(0.0, stock_wh - s_min) / 1000.0 * eta_d
        displaces = min(usable, post_horizon_kwh)
        surplus = max(0.0, usable - post_horizon_kwh)
        return (displaces * replacement + surplus * feed_in) * 1000.0

    print(f"Horizont       : {len(prices)} Slots")
    print(f"Netzpreis      : {min(prices)*1e5:.1f} bis {max(prices)*1e5:.1f} ct/kWh")
    print(f"p_a aktuell    : {current*1e5:.2f} ct/kWh  (forensischer Einstandspreis)")
    print(f"Wiederbeschaffung nach dem Horizont: {replacement*1e5:.2f} ct/kWh")
    print(f"Verbrauch danach, der Energie verwerten kann: {post_horizon_kwh:.1f} kWh")
    print(f"Alles darueber hinaus nur zur Einspeiseverguetung: {feed_in*1e5:.2f} ct/kWh")
    print()

    candidates = [
        ("aktuell (Einstand)", current),
        ("guenstigster Preis im Horizont", min(prices)),
        ("evcc-Vorgabe: min * eta * 0.99", min(prices) * 0.9 * 0.99),
        ("Wiederbeschaffung", replacement),
        ("Wiederbeschaffung + 20 %", replacement * 1.2),
        ("mittlerer Netzpreis", sum(prices) / len(prices)),
    ]

    print("%-32s %8s %9s %9s %10s %11s" % (
        "p_a", "ct/kWh", "SoC Ende", "SoC min", "Netz kWh", "Bewertung"))
    rows = []
    for label, p_a in candidates:
        result = solve(request, p_a)
        soc = result["batteries"][0]["state_of_charge"]
        imports = result["grid_import"]
        grid_kwh = sum(
            value * duration / 3600.0 for value, duration in zip(imports, durations)
        ) / 1000.0
        grid_cost = sum(
            value * duration / 3600.0 * price
            for value, duration, price in zip(imports, durations, prices)
        )
        score = grid_cost - (stock_value(soc[-1]) - stock_value(s_initial))
        rows.append((label, p_a, soc[-1] / capacity * 100, min(soc) / capacity * 100,
                     grid_kwh, score))
        print("%-32s %8.2f %8.1f %% %8.1f %% %9.2f %10.4f EUR" % (
            label, p_a * 1e5, soc[-1] / capacity * 100, min(soc) / capacity * 100,
            grid_kwh, score))

    best = min(rows, key=lambda r: r[5])
    print()
    print(f"Bester Plan nach gemeinsamem Massstab: {best[0]} ({best[1]*1e5:.2f} ct/kWh)")
    baseline = next(r for r in rows if r[0].startswith("aktuell"))
    print(f"Unterschied zum aktuellen Wert: {(baseline[5] - best[5]):+.4f} EUR "
          f"ueber {len(prices)*0.25:.0f} Stunden")


if __name__ == "__main__":
    main()
