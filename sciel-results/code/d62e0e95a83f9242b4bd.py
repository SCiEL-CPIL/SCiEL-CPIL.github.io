"""Improved dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard: keep a small safety margin before the 120s/150s wall-clock limit.
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin

def _elapsed():
    return _time.time() - _START_TIME

def _near_deadline():
    return _elapsed() > 115.0  # conservative: stop heavy work well before limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Strategy:
      1. If near deadline, fall back to a cheap heuristic (max remaining_work).
      2. Otherwise, use a composite score that considers:
         - Total remaining work of the job (longest remaining job first)
         - The minimum duration among machine options for this operation (shortest processing time)
         - The number of remaining operations in the job (fewer remaining ops means closer to completion)
         - A tie-breaker favoring operations that free up machines with less total remaining load

    The idea is a blend of:
      - "Longest Remaining Processing Time" (LRPT) at the job level
      - "Shortest Processing Time" (SPT) at the operation level
      - "Most Urgent" (fewest remaining ops)
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Cheap fallback if we're running out of time
    if _near_deadline():
        max_work = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_work:
                max_work = rw
                best_idx = i
        return int(best_idx)

    # Composite scoring
    scores = []
    for i, op in enumerate(ready_operations):
        job = op.get("job", 0)
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})

        # Minimum duration among available machines (SPT component)
        if machine_options:
            min_dur = min(machine_options.values()) if isinstance(machine_options, dict) else min(machine_options)
        else:
            min_dur = 0

        # Number of remaining operations (urgency: fewer = more urgent)
        urgency = 1.0 / (remaining_ops + 1) if remaining_ops >= 0 else 0.0

        # Composite: weight remaining_work heavily, then min duration inversely, then urgency
        # We want to maximize: remaining_work (long jobs first), minimize min_dur (short ops first), maximize urgency
        # Normalize-ish: use a weighted sum
        score = 1.0 * remaining_work - 0.1 * min_dur + 0.5 * urgency * (remaining_work + 1)

        scores.append(score)

    # Pick the operation with the highest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
