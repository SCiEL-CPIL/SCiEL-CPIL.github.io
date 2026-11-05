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

    Refined heuristic using soft-max surrogate for effective processing time:
    1. Critical-path estimate: remaining_work / soft_min_duration
    2. Soft-max of durations as a smooth surrogate for the bottleneck
    3. Urgency from remaining ops
    4. Flexibility bonus
    """
    work = op["remaining_work"]
    machine_options = op["machine_options"]

    if not machine_options:
        return 0.0

    durations = np.array(list(machine_options.values()), dtype=np.float64)
    min_dur = float(np.min(durations))
    max_dur = float(np.max(durations))
    num_machines = len(durations)

    remaining_ops = op.get("remaining_ops", 0)

    # Soft-min surrogate: smooth approximation of min(durations)
    # Using log-sum-exp with negative values: soft_min = -logsumexp(-d)
    # This is smooth and differentiable, better than hard min for tie-breaking
    neg_d = -durations
    max_neg = np.max(neg_d)
    soft_min = -(max_neg + np.log(np.sum(np.exp(neg_d - max_neg))))

    # Soft-max surrogate for effective bottleneck duration
    # soft_max = logsumexp(d) - log(n) gives a smooth average-max
    max_d = np.max(durations)
    soft_max = max_d + np.log(np.sum(np.exp(durations - max_d)) / num_machines)

    # Criticality: remaining work / soft_min_duration
    # Jobs with high remaining work that can be done on a fast machine are critical
    if soft_min > 1e-9:
        criticality = work / soft_min
    else:
        criticality = work

    # Bottleneck penalty: if the soft-max is much larger than soft-min,
    # this op has high variance in processing time, which is risky
    spread = soft_max - soft_min
    bottleneck_penalty = 0.08 * spread / (soft_min + 1e-9)

    # Weighted composite score
    score = 0.0
    score += 1.0 * criticality          # Primary: critical-path urgency
    score += 0.12 * work                # Stabilizer: raw remaining work
    score += 0.25 / (soft_min + 1e-9)   # Prefer operations that can finish quickly
    score += 0.25 / (remaining_ops + 1e-9)  # Prefer jobs closer to completion (urgency)
    score += 0.04 * num_machines        # Flexibility bonus
    score -= bottleneck_penalty          # Penalize high variance in durations

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
