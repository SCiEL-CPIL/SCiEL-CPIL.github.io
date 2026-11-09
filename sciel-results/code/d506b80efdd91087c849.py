"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill

def _deadline_exceeded():
    return _time.monotonic() >= _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def _soft_min(values, alpha=2.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation of min."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    if len(values) == 1:
        return values[0]
    # Soft min: -1/alpha * log(sum(exp(-alpha * x)))
    neg_alpha_vals = -alpha * values
    max_neg = np.max(neg_alpha_vals)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_alpha_vals - max_neg)))
    return -log_sum / alpha

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Returns index of the ready operation to start next.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, collect machine durations
    all_durations = []
    for op in ready_operations:
        if op["machine_options"]:
            all_durations.append(list(op["machine_options"].values()))
        else:
            all_durations.append([1.0])
    
    # Compute min, max, mean, and soft-min durations
    min_durations = np.array([min(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    max_durations = np.array([max(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    mean_durations = np.array([np.mean(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    
    # Soft-min for smooth duration estimation
    soft_min_durations = np.array([
        _soft_min(np.array(d, dtype=np.float64), alpha=2.0) if d else 1.0 
        for d in all_durations
    ], dtype=np.float64)

    # WSPT ratio: remaining_work / soft_min_duration
    # Higher WSPT means more urgency relative to processing time
    wspt = remaining_work / np.maximum(soft_min_durations, 1e-9)
    
    # Normalize all features
    n_wspt = _normalize(wspt)
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd_min = _normalize(min_durations)
    nd_max = _normalize(max_durations)
    nd_mean = _normalize(mean_durations)
    nd_soft = _normalize(soft_min_durations)

    # Multi-criteria scoring:
    # - WSPT (urgency relative to processing time): higher is better
    # - remaining_work (urgency): higher is better
    # - remaining_ops (keep in motion / flow): higher is better
    # - (1 - nd_soft) for SPT via soft-min: lower duration is better
    # - (1 - nd_max) to avoid operations with high max duration (blocking machines)
    
    # Weights: emphasize WSPT as primary signal, then urgency, flow, SPT, and range penalty
    w_wspt = 0.35      # primary urgency signal
    w_work = 0.20      # secondary urgency
    w_ops = 0.15       # keep in motion
    w_spt = 0.20       # shortest processing time (via soft-min)
    w_range = 0.10     # penalize high max duration (variability/blocking)

    score = (w_wspt * n_wspt + 
             w_work * nw + 
             w_ops * no + 
             w_spt * (1.0 - nd_soft) + 
             w_range * (1.0 - nd_max))

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
