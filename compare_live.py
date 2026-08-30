"""Offline counter-check with a real request read from the running instance.

Runs the bundled solver on the same input with the buffer off and on, and
compares the resulting plans and costs. Nothing is written anywhere and the
live instance is only ever read from.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pytz  # noqa: E402

from src.interfaces.optimization_backends.optimization_backend_local_evopt import (  # noqa: E402
    LocalEVOptBackend,
)

REQ = Path(sys.argv[1] if len(sys.argv) > 1 else "live_req.json")


def run(max_pct, lead=8.0):
    be = LocalEVOptBackend(
        time_frame_base=900,
        time_zone=pytz.timezone("Europe/Berlin"),
        battery_buffer_max_pct=max_pct,
        battery_buffer_lead_hours=lead,
    )
    req = json.loads(REQ.read_text(encoding="utf-8"))
    evopt_req, errors = be._transform_request_from_eos_to_evopt(req)
    if errors:
        print("  Transform-Fehler:", errors)
    s_goal = evopt_req["batteries"][0]["s_goal"] if evopt_req.get("batteries") else []
    resp, _runtime = be.optimize(req, timeout=120)  # optimize returns (response, seconds)
    return evopt_req, s_goal, resp


def summarise(resp):
    if not isinstance(resp, dict):
        return None
    res = resp.get("result", {})
    return (
        res.get("Gesamtkosten_Euro"),
        res.get("akku_soc_pro_stunde") or [],
        res.get("Netzbezug_Wh_pro_Stunde") or [],
    )


def main():
    print(f"Eingabe: {REQ.name}\n")
    rows = []
    for pct in (0, 5, 10, 15):
        evopt_req, s_goal, resp = run(pct)
        nz = [v for v in s_goal if v > 0]
        cost, soc, grid = summarise(resp) or (None, [], [])
        rows.append((pct, s_goal, cost, soc, grid))
        print(f"max_pct={pct:2}: s_goal Slots>0 = {len(nz):3}, "
              f"max = {max(s_goal) if s_goal else 0:.0f} Wh, "
              f"Kosten = {cost if cost is None else round(cost, 3)}")

    base_goal = rows[0][1]
    assert base_goal == [0.0] * len(base_goal), "Aus-Schalter erzeugt kein reines Nullprofil"
    print("\nAus-Schalter: s_goal ist exakt [0.0]*n  OK")

    base_cost = rows[0][2]
    print("\nWirkung auf den Fahrplan:")
    for pct, s_goal, cost, soc, grid in rows:
        if cost is None or base_cost is None:
            print(f"  max_pct={pct:2}: keine Kosten im Response")
            continue
        base_soc = rows[0][3]
        same = base_soc == soc
        # How much higher the plan keeps the battery, and the lowest point it
        # plans to reach - the failure mode the buffer is meant to prevent.
        lift = (sum(soc) / len(soc) - sum(base_soc) / len(base_soc)) if soc and base_soc else 0.0
        print(f"  max_pct={pct:2}: Kosten {cost:8.4f} EUR ({cost - base_cost:+.4f})  "
              f"SoC-Mittel {lift:+.1f} %-Punkte  min SoC {min(soc) if soc else 0:.1f} %  "
              f"identisch: {same}")
        if pct > 0 and same:
            print("     Hinweis: unveraendert - der Strafterm greift hier nicht")


if __name__ == "__main__":
    main()
