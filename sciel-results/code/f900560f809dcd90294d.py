"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 120.0
_START = _time.time()
_MARGIN = 5.0  # seconds to stay under the hard limit

# Lightweight bookkeeping: approximate machine load so we can balance.
# We do not receive completion events, so we accumulate a running estimate
# of how much work has been *assigned* to each machine. This is a proxy
# that helps spread load across machines within a single run.
_MACHINE_LOAD = {}
_LAST_TIME = -1.0

def _machine_load(m):
    return _MACHINE_LOAD.get(m, 0.0)

def _bump_machine_load(m, d):
    _MACHINE_LOAD[m] = _MACHINE_LOAD.get(m, 0.0) + d


def dispatch(ready_operations, time):
    global _LAST_TIME
    
    # Reset state if time goes backwards or resets to 0 (new instance)
    if time < _LAST_TIME or time == 0:
        _MACHINE_LOAD.clear()
    _LAST_TIME = time

    # Deadline guard: if we are about to exceed the hard limit, fall back
    # to the simple baseline (argmax remaining_work) which is O(n) and safe.
    if _time.time() - _START > _DEADLINE - _MARGIN:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # ---- Build per-operation features ---------------------------------
    # job urgency: total remaining work for the job (higher = more urgent)
    # intensity:   work per remaining op (normalizes jobs of different size)
    # own_proc:    minimum processing time over its machine options
    #              (shorter ops first reduces machine idle / blocking)
    # flex:        range of processing times (max - min)
    #              (lower flex = more constrained, should be scheduled earlier)
    urg = []
    inten = []
    own = []
    flex = []
    for op in ready_operations:
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1)) or 1.0
        mo = op.get("machine_options", {})
        # minimum duration among eligible machines
        if mo:
            durs = [float(v) for v in mo.values()]
            own_proc = min(durs)
            max_proc = max(durs)
            flex.append(max_proc - own_proc)
        else:
            own_proc = rw / ro
            flex.append(0.0)
        urg.append(rw)
        inten.append(rw / ro)
        own.append(own_proc)

    urg = np.asarray(urg, dtype=float)
    inten = np.asarray(inten, dtype=float)
    own = np.asarray(own, dtype=float)
    flex = np.asarray(flex, dtype=float)

    # Normalise into [0,1] to make weights comparable.
    def _norm(x):
        rng = x.max() - x.min()
        if rng <= 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / rng

    u = _norm(urg)
    i = _norm(inten)
    o = _norm(own)  # we actually prefer SMALL own_proc, so invert below
    f = _norm(flex)  # we prefer LOW flex (constrained), so invert below

    # ---- Choose the operation -----------------------------------------
    # Combine urgency, intensity, SPT (short processing time), and constraint (low flexibility).
    # Based on observations:
    # - Current program (1.0150) uses w_u=0.40, w_i=0.20, w_o=0.40 (high SPT)
    # - Inspiration 1 (1.0513) uses w_u=0.55, w_i=0.30, w_o=0.15 (high Urgency)
    # - Inspiration 3 (0.9930) added flex with w_u=0.35, w_i=0.15, w_o=0.35, w_f=0.15
    # 
    # The current program's high SPT weight seems to be performing well.
    # Let's try a blend that keeps SPT high but adds a small constraint factor,
    # and slightly reduces urgency to see if the balance improves.
    # We will use: Urgency 0.35, Intensity 0.15, SPT 0.40, Constraint 0.10
    w_u, w_i, w_o, w_f = 0.35, 0.15, 0.40, 0.10
    op_score = w_u * u + w_i * i + w_o * (1.0 - o) + w_f * (1.0 - f)

    best_idx = int(np.argmax(op_score))

    # ---- Choose the machine -------------------------------------------
    # Use a smooth surrogate: minimize processing time primarily, with a
    # small penalty for machine load to break ties and balance work.
    # This is more reliable than load alone since we don't track completions.
    chosen_op = ready_operations[best_idx]
    mo = chosen_op.get("machine_options", {})
    if mo:
        # Compute a score for each machine: duration + small * load
        # The small weight on load ensures we prefer shorter durations
        # but still break ties toward less-loaded machines.
        load_weight = 0.05  # Small weight to avoid distorting the primary criterion
        best_m = None
        best_key = None
        for m, dur in mo.items():
            d = float(dur)
            key = d + load_weight * _machine_load(m)
            if best_key is None or key < best_key:
                best_key = key
                best_m = m
                best_d = d
        if best_m is not None:
            _bump_machine_load(best_m, best_d)
            return (best_idx, best_m)

    # Fallback: just return the index
    return best_idx
# EVOLVE-BLOCK-END
