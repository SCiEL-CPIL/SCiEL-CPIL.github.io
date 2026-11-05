"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0  # safety margin before 120s limit


def _check_deadline():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-compute features for each ready operation
    remaining_works = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        durations = list(op["machine_options"].values())
        remaining_works[i] = rw
        remaining_ops[i] = ro
        min_durations[i] = min(durations) if durations else 0.0

    # Normalize each feature to [0, 1] range (higher is better for scoring)
    rw_min, rw_max = remaining_works.min(), remaining_works.max()
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    md_min, md_max = min_durations.min(), min_durations.max()

    # MWR: more remaining work = higher priority (urgency)
    if rw_max > rw_min:
        mwr_score = (remaining_works - rw_min) / (rw_max - rw_min)
    else:
        mwr_score = np.ones(n, dtype=np.float64)

    # Fewer remaining ops = closer to completion = higher priority
    if ro_max > ro_min:
        few_ops_score = 1.0 - (remaining_ops - ro_min) / (ro_max - ro_min)
    else:
        few_ops_score = np.ones(n, dtype=np.float64)

    # SPT: shorter min duration = less machine blocking = higher priority
    if md_max > md_min:
        spt_score = 1.0 - (min_durations - md_min) / (md_max - md_min)
    else:
        spt_score = np.ones(n, dtype=np.float64)

    # Weighted combination: balance urgency, completion proximity, and efficiency
    # MWR dominates (job urgency), SPT secondary (reduce blocking), few_ops tertiary
    weights = (0.5, 0.3, 0.2)
    combined = weights[0] * mwr_score + weights[1] * spt_score + weights[2] * few_ops_score

    best_idx = int(np.argmax(combined))

    if _check_deadline():
        return 0

    return best_idx
# EVOLVE-BLOCK-END
