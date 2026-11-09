"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level state for deadline guard
_START_TIME = None
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0  # seconds


def _check_deadline():
    """Check if we're approaching the deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME > (_DEADLINE - _SAFETY_MARGIN)


def _soft_min(durations, temperature=1.0):
    """
    Compute a soft minimum of a list of durations using log-sum-exp.
    This provides a smooth surrogate for the hard min that accounts for
    the availability of multiple machines.
    """
    if not durations:
        return 0.0
    vals = np.array(durations, dtype=np.float64)
    min_val = np.min(vals)
    # To avoid overflow, subtract the min
    exp_vals = np.exp((min_val - vals) / temperature)
    log_sum = np.log(np.sum(exp_vals))
    # soft_min = min_val - temperature * log(sum(exp((min_val - vals)/temp)))
    # This approximates the min when temperature is small
    return min_val - temperature * log_sum


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Multi-criteria heuristic combining:
      1. Most remaining work (prioritize long jobs)
      2. Soft-min duration across machines (smooth surrogate for best machine)
      3. Least remaining operations (finish jobs soon to reduce makespan)
    
    We compute a composite score for each ready operation and pick the one
    with the highest priority (lowest score).
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Deadline guard: if we're running low on time, use a simple fast heuristic
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute features for each ready operation
    remaining_works = np.zeros(n)
    soft_min_durations = np.zeros(n)
    remaining_ops = np.zeros(n)
    
    # Temperature for soft-min: smaller = closer to hard min, larger = smoother
    temp = 1.0
    
    for i, op in enumerate(ready_operations):
        durations = op["machine_options"]
        if durations:
            # Use soft-min to account for machine flexibility
            sm_dur = _soft_min(list(durations.values()), temperature=temp)
            soft_min_durations[i] = sm_dur
        else:
            soft_min_durations[i] = 0.0
        remaining_works[i] = op["remaining_work"]
        remaining_ops[i] = op["remaining_ops"]
    
    # Normalize features to [0, 1] range for robust weighting
    rw_min, rw_max = remaining_works.min(), remaining_works.max()
    sd_min, sd_max = soft_min_durations.min(), soft_min_durations.max()
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    
    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    sd_range = sd_max - sd_min if sd_max > sd_min else 1.0
    ro_range = ro_max - ro_min if ro_max > ro_min else 1.0
    
    rw_norm = (remaining_works - rw_min) / rw_range
    sd_norm = (soft_min_durations - sd_min) / sd_range
    ro_norm = (remaining_ops - ro_min) / ro_range
    
    # Composite score: we want to minimize this
    # - Higher remaining_work -> should be prioritized (negative weight)
    # - Lower soft-min duration -> should be prioritized (positive weight for duration)
    # - Lower remaining_ops -> should be prioritized (negative weight)
    
    # Weights: remaining work is most important, then remaining ops, then duration
    w_rw = 1.0
    w_ro = 0.4
    w_sd = 0.2
    
    scores = -w_rw * rw_norm + w_sd * sd_norm - w_ro * ro_norm
    
    # Pick the operation with the lowest score (highest priority)
    idx = int(np.argmin(scores))
    return idx
# EVOLVE-BLOCK-END
