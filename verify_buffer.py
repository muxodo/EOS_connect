"""Local verification of the decaying battery buffer, per the plan.

Runs without touching any live instance: the profile builder is pure arithmetic
over arrays that are already in scope when the optimizer request is assembled.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from interfaces.optimization_backends.optimization_backend_evopt import EVOptBackend  # noqa: E402

CAP = 28000.0
S_MIN = CAP * 0.05
S_MAX = CAP


def backend(max_pct, lead=8.0, tfb=900):
    b = object.__new__(EVOptBackend)
    b.time_frame_base = tfb
    b.battery_buffer_max_pct = max_pct
    b.battery_buffer_lead_hours = lead
    return b


def synth(n=192):
    """One expensive night, a cheap midday block, expensive evening - the shape
    actually measured on this tariff (cheap window at midday, not at night)."""
    price, pv, load = [], [], []
    for t in range(n):
        h = (t * 15 / 60) % 24
        price.append(0.15 if 12 <= h < 15 else 0.33)
        pv.append(3000.0 if 10 <= h < 17 else 0.0)
        load.append(800.0)
    return price, pv, load


def main():
    price, pv, load = synth()
    n = len(price)
    fails = []

    # 1. The off switch must restore the original request exactly.
    off = backend(0)._build_battery_buffer_profile(n, price, pv, load, S_MIN, S_MAX, CAP)
    ok = off == [0.0] * n
    print(f"1. Aus (max_pct=0) -> [0.0]*n exakt: {'OK' if ok else 'FEHLER'}")
    if not ok:
        fails.append("off switch")

    # 2. Profile shape: full buffer far from the checkpoint, zero at it, never above s_max.
    prof = backend(10)._build_battery_buffer_profile(n, price, pv, load, S_MIN, S_MAX, CAP)
    max_buf = CAP * 0.10
    in_range = all(0.0 <= v <= S_MAX + 1e-9 for v in prof)
    never_below_min = all(v == 0.0 or v >= S_MIN - 1e-9 for v in prof)
    # Inside the cheap block the checkpoint has passed, so no surcharge remains.
    cheap_idx = [i for i in range(n) if price[i] <= 0.15]
    # The slot right before a cheap block starts must carry (almost) no buffer.
    starts = [i for i in cheap_idx if i == 0 or price[i - 1] > 0.15]
    at_cp = prof[starts[0]] if starts else None
    print(f"2. Profil: im Bereich [0, s_max] {'OK' if in_range else 'FEHLER'}, "
          f"nie unter s_min {'OK' if never_below_min else 'FEHLER'}")
    print(f"   Puffer am Checkpoint: {at_cp:.0f} Wh (s_min={S_MIN:.0f}), Maximum im Profil: {max(prof):.0f} Wh "
          f"(erwartet <= s_min+{max_buf:.0f})")
    if not (in_range and never_below_min):
        fails.append("profile range")
    if max(prof) > S_MIN + max_buf + 1e-6:
        fails.append("buffer exceeds max_pct")
        print("   FEHLER: Puffer ueberschreitet max_pct")

    # Monotonic decay towards the checkpoint.
    seg = prof[starts[0] - 20:starts[0]] if starts and starts[0] >= 20 else []
    mono = all(seg[i] >= seg[i + 1] - 1e-9 for i in range(len(seg) - 1))
    print(f"   Abschmelzen zum Checkpoint monoton fallend: {'OK' if mono else 'FEHLER'}")
    if not mono:
        fails.append("not monotonic")

    # 3. Edge cases must not raise and must stay well-formed.
    cases = {
        "keine guenstigen Preise": ([0.33] * n, pv, load, CAP, 900),
        "Kapazitaet 0": (price, pv, load, 0.0, 900),
        "n=1": ([0.33], [0.0], [800.0], CAP, 900),
        "Stundenraster": (price[:48], pv[:48], load[:48], CAP, 3600),
    }
    for name, (pr, p_, l_, cap, tfb) in cases.items():
        try:
            r = backend(10, tfb=tfb)._build_battery_buffer_profile(
                len(pr), pr, p_, l_, cap * 0.05, cap, cap
            )
            good = len(r) == len(pr) and all(isinstance(v, float) for v in r)
            print(f"3. Randfall {name:24}: {'OK' if good else 'FEHLER'} (len={len(r)})")
            if not good:
                fails.append(name)
        except Exception as exc:  # noqa: BLE001 - any raise is a failure here
            print(f"3. Randfall {name:24}: FEHLER {type(exc).__name__}: {exc}")
            fails.append(name)

    # 4. lead_hours must change the shape. Direction matters and is easy to get
    # backwards: frac = min(1, hours_left / lead), so a SHORT lead saturates
    # sooner and therefore reserves more overall; a long lead ramps gently.
    short = backend(10, lead=2.0)._build_battery_buffer_profile(n, price, pv, load, S_MIN, S_MAX, CAP)
    long_ = backend(10, lead=16.0)._build_battery_buffer_profile(n, price, pv, load, S_MIN, S_MAX, CAP)
    differs = short != long_ and sum(short) > sum(long_)
    print(f"4. lead_hours wirkt (2h saettigt frueher, reserviert mehr als 16h): "
          f"{'OK' if differs else 'FEHLER'}  (Summe 2h={sum(short):.0f}, 16h={sum(long_):.0f})")
    if not differs:
        fails.append("lead_hours ineffective")

    # 5. max_pct must scale the reserve proportionally.
    p5 = backend(5)._build_battery_buffer_profile(n, price, pv, load, S_MIN, S_MAX, CAP)
    p20 = backend(20)._build_battery_buffer_profile(n, price, pv, load, S_MIN, S_MAX, CAP)
    above5 = sum(v - S_MIN for v in p5 if v > 0)
    above20 = sum(v - S_MIN for v in p20 if v > 0)
    scales = abs(above20 / above5 - 4.0) < 0.01 if above5 > 0 else False
    print(f"5. max_pct skaliert linear (20%% = 4x von 5%%): {'OK' if scales else 'FEHLER'} "
          f"(Verhaeltnis {above20 / above5:.2f})" if above5 > 0 else "5. FEHLER: 5%% reserviert nichts")
    if not scales:
        fails.append("max_pct scaling")

    print()
    print("ERGEBNIS:", "alle Pruefungen bestanden" if not fails else f"FEHLGESCHLAGEN: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
