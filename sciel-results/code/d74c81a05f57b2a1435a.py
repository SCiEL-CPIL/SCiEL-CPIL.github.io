"""Dispatch rule using adaptive rank-based priority with context-aware weighting."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE


def _rank(arr):
    """Compute ranks (0 to n-1) where 0 is smallest, n-1 is largest."""
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.arange(len(arr))
    return ranks


def _soft_min(arr, temperature=1.0):
    """
    Smooth surrogate for min using log-sum-exp with negative sign.
    soft_min(x) = -log(sum(exp(-x_i / T))) * T
    As T -> 0, this approaches the true min.
    """
    neg_arr = -arr / temperature
    # For numerical stability
    max_neg = np.max(neg_arr)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_arr - max_neg)))
    return -log_sum * temperature


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Quick deadline guard
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each op, compute min duration and soft-min duration
    min_durations = np.empty(n, dtype=np.float64)
    all_durations = []
    machine_sets = []
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            min_durations[i] = 1.0
            all_durations.append([1.0])
            machine_sets.append(set())
        else:
            durations = list(mo.values())
            min_durations[i] = min(durations)
            all_durations.append(durations)
            machine_sets.append(set(mo.keys()))

    # Compute soft-min for each operation (smooth surrogate for min duration)
    # Use a small temperature to approximate min while remaining differentiable
    soft_min_durations = np.empty(n, dtype=np.float64)
    for i, durs in enumerate(all_durations):
        if len(durs) == 1:
            soft_min_durations[i] = durs[0]
        else:
            soft_min_durations[i] = _soft_min(np.array(durs, dtype=np.float64), temperature=0.5)

    # Critical path feature: remaining work per remaining operation
    # This identifies operations that are on the critical path (high work per op)
    critical_path = np.where(remaining_ops > 0, remaining_work / np.maximum(remaining_ops, 1), remaining_work)

    # WSPT ratio: remaining_work / min_duration (work per unit time urgency)
    wspt = np.where(min_durations > 0, remaining_work / np.maximum(min_durations, 1e-9), remaining_work)

    # Compute ranks for each feature (0 = lowest, n-1 is highest)
    rank_work = _rank(remaining_work)       # higher = more remaining work
    rank_ops = _rank(remaining_ops)          # higher = more ops left
    rank_cp = _rank(critical_path)           # higher = more critical (more work per op)
    rank_wspt = _rank(wspt)                  # higher = more urgent (more work per time)
    rank_spt = _rank(soft_min_durations)     # higher = longer duration (we want lower)
    rank_spt_inv = (n - 1) - rank_spt        # invert: higher = shorter duration (better SPT)

    # Machine load balancing: count how many ready ops can use each machine
    machine_load = {}
    for i in range(n):
        for m in machine_sets[i]:
            machine_load[m] = machine_load.get(m, 0) + 1

    # For each op, find the least-loaded machine it can use
    # and compute slack = (max_load - min_load_for_op) as a proxy for contention reduction
    max_load = max(machine_load.values()) if machine_load else 1
    slack_scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        if machine_sets[i]:
            min_load = min(machine_load[m] for m in machine_sets[i])
            slack_scores[i] = max_load - min_load  # higher = more slack available
        else:
            slack_scores[i] = 0.0

    rank_slack = _rank(slack_scores)  # higher = more slack (less contention)

    # Context-aware weighting with slack component:
    # - When few ops are ready (n <= 3), use more SPT to free machines quickly
    # - When many ops are ready (n >= 6), use more critical path + slack to avoid bottlenecks
    # - Middle range: balanced approach
    if n <= 3:
        w_work = 0.20
        w_ops = 0.10
        w_cp = 0.10
        w_wspt = 0.15
        w_spt = 0.35
        w_slack = 0.10
    elif n <= 5:
        w_work = 0.25
        w_ops = 0.10
        w_cp = 0.15
        w_wspt = 0.15
        w_spt = 0.25
        w_slack = 0.10
    else:
        w_work = 0.25
        w_ops = 0.10
        w_cp = 0.20
        w_wspt = 0.15
        w_spt = 0.15
        w_slack = 0.15

    # Rank-based score: combine ranked features with context-aware weights
    score = (w_work * rank_work + w_ops * rank_ops + w_cp * rank_cp + 
             w_wspt * rank_wspt + w_spt * rank_spt_inv + w_slack * rank_slack)

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
