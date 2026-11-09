"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard to ensure we finish well within the 120s evaluation limit
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds buffer before hard limit


def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def _soft_min(values, beta=1.0):
    """
    Compute a soft minimum using log-sum-exp.
    soft_min = -log(sum(exp(-beta * v))) / beta
    As beta -> inf, this approaches the true min.
    For moderate beta, it captures the distribution of machine options.
    """
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    # Numerically stable log-sum-exp
    neg_beta_v = -beta * arr
    max_val = np.max(neg_beta_v)
    if np.isinf(max_val):
        return float(np.min(arr))
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_v - max_val)))
    return -log_sum / beta


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a hybrid heuristic with soft-min for machine durations to capture
    the full distribution of processing times, not just the minimum.
    """
    if not ready_operations:
        return 0
    
    # Check if we're approaching the deadline - if so, use a fast fallback
    if _check_deadline():
        # Fast fallback: pick the operation with the most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    
    # Calculate soft-min durations from machine_options
    # beta=2.0 balances between true min and mean, capturing flexibility
    soft_min_durations = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        if machine_options:
            vals = list(machine_options.values())
            sm = _soft_min(vals, beta=2.0)
        else:
            sm = 0.0
        soft_min_durations.append(sm)
    
    # Also compute true min for reference
    true_min_durations = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        if machine_options:
            true_min_durations.append(min(machine_options.values()))
        else:
            true_min_durations.append(0)
    
    # Extract features
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops_counts = [op["remaining_ops"] for op in ready_operations]
    
    # Heuristic 1: Most work remaining (MW) - captures urgency
    mw_scores = remaining_works
    
    # Heuristic 2: Soft-min duration (LPT-like) - captures processing time distribution
    lpt_scores = soft_min_durations
    
    # Heuristic 3: Most operations remaining (MOR)
    mor_scores = remaining_ops_counts
    
    # Heuristic 4: Work density = remaining_work / (remaining_ops + 1)
    # High density means each remaining op is long, so scheduling it sooner helps
    work_density = []
    for i in range(n):
        ops = remaining_ops_counts[i] + 1  # avoid division by zero
        work_density.append(remaining_works[i] / ops)
    
    # Heuristic 5: Flexibility penalty - ops with few machine options are more constrained
    # Lower flexibility (fewer machines) should be scheduled sooner
    flexibility_scores = []
    for op in ready_operations:
        n_machines = len(op["machine_options"])
        # Invert: fewer machines -> higher score (more urgent)
        flexibility_scores.append(1.0 / max(n_machines, 1))
    
    # Normalize each feature to [0, 1]
    def normalize(scores):
        mx = max(scores) if scores else 1
        if mx <= 0:
            return [0.0] * len(scores)
        return [x / mx for x in scores]
    
    norm_mw = normalize(mw_scores)
    norm_lpt = normalize(lpt_scores)
    norm_mor = normalize(mor_scores)
    norm_wd = normalize(work_density)
    norm_flex = normalize(flexibility_scores)
    
    # Combine with tuned weights:
    # MW (0.30) - primary urgency signal
    # LPT soft-min (0.25) - processing time distribution
    # Work density (0.20) - per-op urgency
    # MOR (0.15) - depth of remaining work
    # Flexibility (0.10) - constrained ops first
    w_mw, w_lpt, w_wd, w_mor, w_flex = 0.30, 0.25, 0.20, 0.15, 0.10
    
    combined_scores = [
        w_mw * mw + w_lpt * lpt + w_wd * wd + w_mor * mor + w_flex * flex
        for mw, lpt, wd, mor, flex in zip(
            norm_mw, norm_lpt, norm_wd, norm_mor, norm_flex
        )
    ]
    
    # Return index of the operation with the highest combined score
    return int(np.argmax(combined_scores))
# EVOLVE-BLOCK-END
