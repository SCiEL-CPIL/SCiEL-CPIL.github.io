"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Track deadline to avoid timeout; 120s limit, keep safety margin
_DEADLINE = _time.monotonic() + 115.0


def _min_duration(op):
    """Minimum processing time across machine options."""
    return min(op["machine_options"].values())


def _max_duration(op):
    """Maximum processing time across machine options."""
    return max(op["machine_options"].values())


def _mean_duration(op):
    """Mean processing time across machine options."""
    return float(np.mean(list(op["machine_options"].values())))


def dispatch(ready_operations, time):
    """
    Dispatch rule: composite heuristic combining:
    - Longest processing time (LPT) via min_duration (inflexibility penalty)
    - Remaining work (total load)
    - Job completion proximity (fewer remaining ops = closer to done)
    
    We use a weighted composite score. Higher score = more urgent to start.
    
    The idea: prioritize operations that are both long (to avoid late bottlenecks)
    and belong to jobs that still have significant remaining work (to keep pipeline
    moving). Operations with few remaining ops get a small boost to finish jobs.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Compute features for each ready operation
    scores = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        min_dur = _min_duration(op)
        max_dur = _max_duration(op)
        mean_dur = _mean_duration(op)
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # LPT component: prefer longer min-duration (more inflexible)
        lpt_score = min_dur
        
        # Remaining work component: prefer more total work left
        rw_score = remaining_work
        
        # Completion proximity: jobs with fewer remaining ops are closer to done
        # Give a small bonus to finish them (reduces tail latency)
        # Normalize: fewer remaining ops -> higher bonus
        completion_bonus = 1.0 / (1.0 + remaining_ops)
        
        # Inflexibility: operations with narrow machine choice ranges are more constrained
        # If min_dur == max_dur, the operation is fixed to one duration -> more critical
        inflexibility = 1.0 - (max_dur - min_dur) / (max_dur + 1e-9)
        
        # Composite score
        # Weights: LPT is primary, remaining work secondary, completion small
        scores[i] = (
            1.0 * lpt_score +
            0.5 * rw_score / (max(rw_score, 1.0)) * min_dur +  # scaled remaining work
            0.3 * completion_bonus * min_dur +
            0.2 * inflexibility * min_dur
        )
    
    # Safety: check deadline
    if _time.monotonic() > _DEADLINE - 5.0:
        # Fallback: just pick max min_duration
        min_durs = np.array([_min_duration(op) for op in ready_operations])
        return int(np.argmax(min_durs))
    
    idx = int(np.argmax(scores))
    return idx
# EVOLVE-BLOCK-END
