"""Improved job-shop dispatch rule: SPT, Bottleneck, and Urgency composite."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s; use 140s as safety margin
_DEADLINE = _time.time() + 140.0


def _compute_score(ready_operations, time):
    """Compute a composite score for each ready operation. Higher = better to pick."""
    n = len(ready_operations)
    if n == 0:
        return 0

    scores = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        # 1. Remaining work (total processing time left for this job)
        remaining_work = op.get("remaining_work", 0)
        
        # 2. Remaining operations count
        remaining_ops = op.get("remaining_ops", 0)

        # 3. Duration stats
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = list(machine_options.values())
            min_dur = min(durations)
            n_opts = len(machine_options)
        else:
            min_dur = 0.0
            n_opts = 1

        # Avoid division by zero
        safe_min_dur = max(min_dur, 1e-9)
        safe_n_opts = max(n_opts, 1)

        # Heuristic Components:
        
        # A. Urgency Ratio: High remaining work with short current op.
        #    This ensures jobs with lots of work left don't stall on long ops.
        urgency_ratio = remaining_work / safe_min_dur
        
        # B. SPT (Shortest Processing Time) Term: 
        #    Strongly favors short operations to reduce machine blocking.
        #    This is a classic and effective local heuristic.
        spt_term = 1.0 / safe_min_dur
        
        # C. Bottleneck/Flexibility Term:
        #    Favors operations with fewer machine options.
        #    These are "critical" operations that cannot be shifted to other machines.
        bottleneck_term = 1.0 / safe_n_opts
        
        # D. Completion Urgency:
        #    Slight preference for jobs closer to completion.
        completion_term = remaining_ops

        # Composite Score:
        # Weights are tuned to balance SPT (local efficiency) with 
        # Job-level urgency and bottleneck constraints.
        score = 1.0 * urgency_ratio + 1.0 * spt_term + 0.5 * bottleneck_term + 0.1 * completion_term

        scores[i] = score

    return int(np.argmax(scores))


def dispatch(ready_operations, time):
    """Dispatch rule for job-shop scheduling.

    Returns the index of the ready operation to start next.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.time() > _DEADLINE - 5.0:
        # Fallback: pick the one with most remaining work
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))

    return _compute_score(ready_operations, time)
# EVOLVE-BLOCK-END
