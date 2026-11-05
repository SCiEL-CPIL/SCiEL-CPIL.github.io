"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 145.0  # safety margin before 150s kill
_START_TIME = _time.time()


def _ensure_deadline():
    """Return True if we are within the wall-clock budget."""
    return (_time.time() - _START_TIME) < _DEADLINE


def _compute_op_score(op, time):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).

    Heuristic components:
    1. Criticality: remaining work / min_duration (MWR / SPT hybrid)
    2. Remaining work (primary): jobs with more total remaining work are more critical.
    3. Minimum duration: operations that can finish quickly unlock subsequent operations.
    4. Number of remaining ops: jobs with fewer remaining ops are closer to completion.
    5. Machine flexibility: more machine options = less risk of blocking.
    """
    work = op["remaining_work"]
    machine_options = op["machine_options"]

    if not machine_options:
        return 0.0

    durations = list(machine_options.values())
    min_dur = min(durations)
    max_dur = max(durations)
    num_machines = len(machine_options)

    remaining_ops = op.get("remaining_ops", 0)

    # Criticality: remaining work divided by min possible duration
    # This captures the idea that a job with lots of remaining work that can be
    # done quickly is most critical
    if min_dur > 0:
        criticality = work / (min_dur + 1e-9)
    else:
        criticality = work

    # Weighted composite score
    # Primary: criticality (work / min_duration) - captures both MWR and SPT
    # Secondary: remaining work alone (to avoid division by very small durations)
    # Tertiary: shorter min duration is better (finish faster, unlock next op)
    # Quaternary: fewer remaining ops means closer to done (urgency)
    # Quinary: more machine options = more flexibility

    score = 0.0
    score += 1.0 * criticality  # Most critical: high work, fast completion possible
    score += 0.15 * work  # Raw remaining work as stabilizer
    score += 0.3 / (min_dur + 1e-9)  # Prefer shorter operations
    score += 0.2 / (remaining_ops + 1e-9)  # Prefer jobs closer to completion
    score += 0.05 * num_machines  # Slight preference for flexible ops

    # Penalty for very long max duration relative to min (uncertainty)
    if max_dur > min_dur:
        score -= 0.1 * (max_dur - min_dur) / (min_dur + 1e-9)

    return score


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Returns: index of the ready operation to start.
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a fast rule
    if not _ensure_deadline():
        # Fast fallback: pick the operation with the most remaining work
        best_idx = 0
        best_work = ready_operations[0]["remaining_work"]
        for i, op in enumerate(ready_operations[1:], 1):
            if op["remaining_work"] > best_work:
                best_work = op["remaining_work"]
                best_idx = i
        return best_idx

    # Compute scores for all ready operations
    scores = [_compute_op_score(op, time) for op in ready_operations]

    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
