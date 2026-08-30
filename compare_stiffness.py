"""Does the goal-penalty scale actually turn the buffer from an obligation into
an economic trade-off? Run against a real request read from the live instance.

Expectation: at scale 1.0 the optimiser honours the buffer at almost any cost,
and as the scale falls it should start trading it away - less grid energy bought,
cost approaching the no-buffer baseline, while still lifting the SOC trough
somewhat where that is cheap.
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
TZ = pytz.timezone("Europe/Berlin")


def run(max_pct, scale=1.0, lead=8.0):
    be = LocalEVOptBackend(
        time_frame_base=900,
        time_zone=TZ,
        battery_buffer_max_pct=max_pct,
        battery_buffer_lead_hours=lead,
        battery_buffer_penalty_scale=scale,
    )
    req = json.loads(REQ.read_text(encoding="utf-8"))
    resp, _ = be.optimize(req, timeout=180)
    res = resp["result"]
    soc = res.get("akku_soc_pro_stunde") or []
    grid = res.get("Netzbezug_Wh_pro_Stunde") or []
    return res.get("Gesamtkosten_Euro"), soc, sum(grid) / 1000.0


def main():
    base_cost, base_soc, base_grid = run(0)
    print(f"Basis (Puffer aus): {base_cost:.4f} EUR, Netzbezug {base_grid:.1f} kWh, "
          f"min SoC {min(base_soc):.2f} %\n")
    print(f"{'Steifigkeit':>12}{'Kosten':>10}{'Diff':>9}{'Netz kWh':>10}{'min SoC':>9}")
    for scale in (1.0, 0.1, 0.05, 0.01, 0.005, 0.001):
        cost, soc, grid = run(20, scale=scale)
        print(f"{scale:>12}{cost:10.4f}{cost - base_cost:+9.4f}{grid:10.1f}{min(soc):8.2f}%")


if __name__ == "__main__":
    main()
