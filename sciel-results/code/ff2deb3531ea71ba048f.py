"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def _soft_min(arr, temperature=1.0):
    """Compute soft minimum using log-sum-exp for smoothness."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    # Soft min = -soft_max(-arr)
    neg_arr = -arr
    max_neg = np.max(neg_arr)
    if max_neg == -np.inf:
        return 0.0
    shifted = neg_arr - max_neg
    log_sum = np.log(np.sum(np.exp(shifted / temperature))) * temperature
    soft_max = max_neg + log_sum
    return -soft_max

def dispatch(ready_operations, time):
    if _deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, compute min and mean duration among its machine options
    min_durations = np.zeros(n, dtype=np.float64)
    mean_durations = np.zeros(n, dtype=np.float64)
    n_options = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if opts:
            vals = np.array(list(opts.values()), dtype=np.float64)
            min_durations[i] = np.min(vals)
            mean_durations[i] = np.mean(vals)
            n_options[i] = len(opts)
        else:
            min_durations[i] = 1.0
            mean_durations[i] = 1.0
            n_options[i] = 1.0

    # WSPT ratio: remaining_work / min_duration
    wspt = remaining_work / np.maximum(min_durations, 1e-9)
    
    # Bottleneck risk: operations with few options and high mean duration
    # are more likely to cause machine starvation
    # Use soft-min to smoothly capture the "best" machine option
    # Low flexibility = low n_options, high processing time = high mean_duration
    
    # Flexibility score: higher when more options available
    flexibility = _normalize(n_options)
    
    # Processing time pressure: higher when mean duration is high relative to min
    # This captures inefficiency when forced to use slower machines
    duration_spread = (mean_durations - min_durations) / np.maximum(mean_durations, 1e-9)
    duration_pressure = _normalize(duration_spread)
    
    # Bottleneck risk: combine low flexibility with high duration pressure
    # Use multiplicative interaction as suggested by knowledge
    bottleneck_risk = (1.0 - flexibility) * (0.5 + 0.5 * duration_pressure)
    n_bottleneck = _normalize(bottleneck_risk)
    
    # Normalize features
    n_wspt = _normalize(wspt)
    no = _normalize(remaining_ops)
    nd = _normalize(min_durations)

    # We want to MAXIMIZE a score that favors:
    # - Higher WSPT (urgency - more work relative to current op)
    # - Higher remaining_ops (keep in motion)
    # - Lower min_duration (SPT) -> so we use 1 - nd
    # - Higher bottleneck risk (prioritize operations that could cause starvation)
    
    # Contextual weighting: emphasize bottleneck risk more when there are
    # many ready operations (more contention)
    w_wspt = 0.4
    w_ops = 0.25
    w_spt = 0.15
    w_bottleneck = 0.20

    score = w_wspt * n_wspt + w_ops * no + w_spt * (1.0 - nd) + w_bottleneck * n_bottleneck

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
