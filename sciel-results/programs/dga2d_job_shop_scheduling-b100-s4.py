"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


def dispatch(ready_operations, time):
    # Check deadline guard
    if _time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick the operation with minimum processing time on any machine
        min_durations = [min(op["machine_options"].values()) for op in ready_operations]
        return int(np.argmin(min_durations))
    
    if not ready_operations:
        return 0
    
    # Use soft-min for smooth surrogate of minimum duration
    # and soft-max for smooth surrogate of maximum remaining work
    
    n = len(ready_operations)
    rw_list = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=float)
    min_dur_list = np.array([min(op["machine_options"].values()) if op["machine_options"] else 0 for op in ready_operations], dtype=float)
    rem_ops_list = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=float)
    
    # Soft-min for processing times (lower is better, so we want to minimize)
    # Use log-sum-exp trick for numerical stability
    # soft_min(x) = -logsumexp(-x)
    def soft_min(x, tau=1.0):
        if len(x) == 0:
            return 0.0
        neg_x = -x / tau
        max_neg_x = np.max(neg_x)
        if max_neg_x > 700:  # numerical safety
            return float(np.min(x))
        log_sum = np.log(np.sum(np.exp(neg_x - max_neg_x))) + max_neg_x
        return -log_sum * tau
    
    # Soft-max for remaining work (higher is better)
    def soft_max(x, tau=1.0):
        if len(x) == 0:
            return 0.0
        x_scaled = x / tau
        max_x = np.max(x_scaled)
        if max_x > 700:  # numerical safety
            return float(np.max(x))
        log_sum = np.log(np.sum(np.exp(x_scaled - max_x))) + max_x
        return log_sum * tau
    
    # Compute soft-min for each operation's machine options
    # This gives a smooth estimate of the minimum processing time
    # Using a smaller tau for a sharper approximation of the minimum
    soft_min_durs = np.array([soft_min(np.array(list(op["machine_options"].values()), dtype=float), tau=0.1) for op in ready_operations])
    
    # SPT term: 1/duration favors short operations on specific machines (local efficiency)
    # Use reciprocal with a small epsilon to avoid division by zero
    eps = 1e-6
    spt_term = 1.0 / (soft_min_durs + eps)
    
    # Combine scores with smooth surrogates
    # Balanced strategy: global job urgency (remaining work) + local machine efficiency (SPT) + bottleneck awareness
    
    # Normalize components
    rw_range = np.max(rw_list) - np.min(rw_list) if np.max(rw_list) > np.min(rw_list) else 1.0
    dur_range = np.max(soft_min_durs) - np.min(soft_min_durs) if np.max(soft_min_durs) > np.min(soft_min_durs) else 1.0
    ops_range = np.max(rem_ops_list) - np.min(rem_ops_list) if np.max(rem_ops_list) > np.min(rem_ops_list) else 1.0
    spt_range = np.max(spt_term) - np.min(spt_term) if np.max(spt_term) > np.min(spt_term) else 1.0
    
    rw_norm = (rw_list - np.min(rw_list)) / rw_range
    dur_norm = (soft_min_durs - np.min(soft_min_durs)) / dur_range
    ops_norm = (rem_ops_list - np.min(rem_ops_list)) / ops_range
    spt_norm = (spt_term - np.min(spt_term)) / spt_range
    
    # Score: balance global urgency (remaining work), local efficiency (SPT), and bottleneck awareness
    # SPT term gets strong weight to prioritize short operations on specific machines
    # Remaining work provides global job completion pressure
    # Bottleneck (high duration) term provides secondary awareness for long operations
    scores = rw_norm * 0.8 + spt_norm * 1.2 + dur_norm * 0.3 + ops_norm * 0.2
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
