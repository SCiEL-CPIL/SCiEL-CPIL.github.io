"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    # Deadline guard: keep a safety margin before the 150s limit.
    # We track elapsed time globally to avoid pathological loops.
    if not hasattr(dispatch, "_t0"):
        dispatch._t0 = _time.monotonic()
    if _time.monotonic() - dispatch._t0 > 140.0:
        # Safety fallback: return the first ready op.
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # For each ready operation, compute:
    #   1) fastest machine duration (min over machine_options)
    #   2) remaining_work (total work left for the job)
    #   3) flexibility (number of machine options)
    #   4) remaining_ops (number of ops left for the job)

    # Strategy: combine multiple heuristics into a single score.
    # We want to prioritize operations that:
    #   - Have high remaining work (long jobs first, like WSPT)
    #   - Have short processing times on their fastest machine (SPT)
    #   - Have high flexibility (more machine options = more scheduling freedom)
    #   - Are from jobs with fewer remaining ops (to avoid bottlenecks)

    # Extract features
    durations = np.empty(n)
    remaining_work = np.empty(n)
    flexibility = np.empty(n)
    remaining_ops = np.empty(n)

    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        durs = list(mo.values())
        durations[i] = min(durs)
        remaining_work[i] = op["remaining_work"]
        flexibility[i] = len(mo)
        remaining_ops[i] = op["remaining_ops"]

    # Normalize features to [0, 1] range for stable weighting
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    norm_dur = _norm(durations)          # lower is better
    norm_rw = _norm(remaining_work)      # higher is better
    norm_flex = _norm(flexibility)       # higher is better
    norm_ro = _norm(remaining_ops)       # lower is better (fewer remaining ops = closer to done)

    # Weighted score: we want high score = pick this op
    # - High remaining work is good (WSPT intuition)
    # - Short duration is good (SPT intuition)
    # - High flexibility is good
    # - Fewer remaining ops is good (finish jobs quickly)

    w_rw = 0.35
    w_dur = 0.30
    w_flex = 0.15
    w_ro = 0.20

    score = (w_rw * norm_rw
             + w_dur * (1.0 - norm_dur)
             + w_flex * norm_flex
             + w_ro * (1.0 - norm_ro))

    idx = int(np.argmax(score))

    # Return (index, machine) to let the simulator place on the fastest machine
    mo = ready_operations[idx]["machine_options"]
    best_machine = min(mo, key=mo.get)

    return (idx, best_machine)
# EVOLVE-BLOCK-END
