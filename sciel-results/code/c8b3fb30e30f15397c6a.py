"""Open-shop dispatch: minimize makespan via a smooth lower-bound surrogate."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

_DEADLINE = 115.0  # safety margin below the 120 s wall-clock limit
_START = _time.monotonic()
_BETA = 6.0        # soft-max sharpness (higher -> closer to hard max)


def _now():
    return _time.monotonic()


def _soft_max(vals, beta):
    """Smooth max via log-sum-exp; falls back to plain max for tiny beta."""
    if beta <= 1e-6:
        return float(max(vals))
    m = max(vals)
    return m + math.log(sum(math.exp((v - m) * beta) for v in vals)) / beta


def _min_machine_load(op):
    """Minimum remaining machine load among the op's machine options."""
    mo = op.get("machine_options")
    if not mo:
        return 0.0
    return min(mo.get(m, 0.0) for m in mo)


def _job_critical_path(op):
    """Lower bound on the job's completion time = sum of remaining op durations."""
    ro = op.get("remaining_ops")
    if not ro:
        return 0.0
    tot = 0.0
    for r in ro:
        mo = r.get("machine_options")
        if mo:
            tot += min(mo.values())
    return tot


def dispatch(ready_operations, time):
    # Deadline guard: if we are dangerously close to the limit, fall back fast.
    if _now() - _START > _DEADLINE:
        return int(np.argmax([op.get("remaining_work", 0.0) for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Precompute per-operation features once.
    cpl = [0.0] * n
    minload = [0.0] * n
    spt = [0.0] * n
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options") or {}
        if mo:
            spt[i] = min(mo.values())
        cpl[i] = _job_critical_path(op)
        minload[i] = _min_machine_load(op)

    best_idx = 0
    best_score = float("inf")
    for i in range(n):
        # After starting op i, the makespan lower bound is driven by:
        #   - job i's remaining critical path (cpl[i] - its op duration)
        #   - the machine it occupies (minload[i])
        # plus the other jobs' critical paths (they run concurrently).
        job_after = max(cpl[i] - spt[i], 0.0)
        lower_bounds = [job_after, minload[i]] + [cpl[j] for j in range(n) if j != i]
        score = _soft_max(lower_bounds, _BETA)
        # Tie-break: prefer shorter processing time (SPT) to free the machine.
        score += 1e-4 * spt[i]
        if score < best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
