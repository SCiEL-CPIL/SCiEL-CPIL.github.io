"""Rewritten open-shop dispatch rule with a composite urgency/duration score."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Safety margin before the 150s wall-clock kill; track per-call start.
_DEADLINE_SECONDS = 145.0
_call_start = None


def dispatch(ready_operations, time):
    global _call_start
    if _call_start is None:
        _call_start = _time.monotonic()
    if _time.monotonic() - _call_start > _DEADLINE_SECONDS:
        # Safety margin: return the first ready operation.
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Build per-operation features.
    remaining_work = np.empty(n, dtype=float)
    min_dur = np.empty(n, dtype=float)
    for i, op in enumerate(ready_operations):
        remaining_work[i] = float(op["remaining_work"])
        opts = op["machine_options"]
        min_dur[i] = float(min(opts.values())) if opts else 0.0

    # Composite score: favor high remaining work (job criticality) and
    # short minimum duration (reduces makespan / SPT effect).
    # Use a smooth, scale-robust urgency density: work / (min_dur + eps).
    eps = 1e-9
    score = remaining_work / (min_dur + eps)

    # Deadline guard: if computation is taking too long, fall back to a
    # simple rule rather than risk a timeout.
    if _time.monotonic() - _call_start > _DEADLINE_SECONDS:
        return 0

    idx = int(np.argmax(score))
    return idx
# EVOLVE-BLOCK-END
