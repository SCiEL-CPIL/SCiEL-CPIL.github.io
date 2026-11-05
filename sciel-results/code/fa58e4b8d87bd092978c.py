"""Open-shop scheduling dispatch rule: operation-based LPT with job-work tie-breaking."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Safety deadline guard: the evaluator kills after 150s wall-clock.
# We track cumulative time spent in dispatch to bail out gracefully.
_START_TIME = _time.monotonic()
_TIME_BUDGET = 120.0  # seconds; leave margin before 150s kill
_elapsed = [0.0]  # mutable container to persist across calls


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Operation-based Longest Processing Time (LPT) with job-work tie-breaking.
    - Primary: maximize the duration of the operation on its assigned machine (LPT)
    - Secondary: maximize remaining work of the job (to finish critical jobs)
    - Tertiary: minimize remaining ops (to reduce tail latency)
    
    For each ready operation, we consider the minimum duration across its machine_options
    (since in open-shop, the operation can be assigned to any machine, and we want to
    prioritize the operation that will take the longest even on its best machine, or
    alternatively, the one with the longest processing time to reduce idle time).
    
    Actually, in open-shop scheduling, each operation must be processed on a specific machine.
    The machine_options dict gives {machine: duration}. The operation will be assigned to
    one of these machines. A common heuristic is to pick the operation with the largest
    processing time (on its assigned machine) to keep busy machines busy.
    
    Since we don't know which machine will be chosen, we use the maximum duration across
    machine_options as a proxy for "how long this operation will keep a machine busy",
    or the minimum duration if we assume the best machine is chosen.
    
    Let's use: primary key = max duration in machine_options (LPT on the operation),
    secondary key = remaining_work of the job,
    tertiary key = -remaining_ops (fewer remaining ops preferred).
    """
    # Deadline guard
    current = _time.monotonic()
    if current - _START_TIME > _TIME_BUDGET:
        # Fallback: just return the first ready operation
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Compute scores for each ready operation
    scores = []
    for i, op in enumerate(ready_operations):
        # Get machine options and compute max duration (LPT proxy)
        machine_opts = op.get("machine_options", {})
        if machine_opts:
            max_dur = max(machine_opts.values())
            min_dur = min(machine_opts.values())
            # Use max duration as primary: this operation will keep a machine busy the longest
            # This reduces idle time on machines
        else:
            max_dur = 0
            min_dur = 0

        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)

        # Primary: max duration (LPT)
        # Secondary: remaining work of the job
        # Tertiary: -remaining ops (fewer remaining is better, so negate)
        score = (max_dur, remaining_work, -remaining_ops)
        scores.append(score)

    # Find the index with the maximum score (lexicographic)
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
