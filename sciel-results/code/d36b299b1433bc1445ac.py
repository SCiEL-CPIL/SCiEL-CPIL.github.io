"""Dispatch rule for open-shop scheduling: choose ready operation with best priority."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: track start time for safety
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def _elapsed_ok():
    """Check if we're still within safe time budget."""
    return (_time.time() - _START_TIME) < (150.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.
    
    Strategy: Combine multiple heuristics to minimize makespan:
    1. Most remaining work (MRW) - prioritize jobs with more work left
    2. Smooth LPT via soft-min/soft-max of machine durations
    3. Remaining operations count - keep pipeline full
    4. Time-aware urgency: operations with less slack relative to current time
    5. Downstream unlock: prefer ops that free up more remaining ops
    
    We use a weighted combination with smooth surrogates for robustness.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Smooth soft-min and soft-max of machine option durations via log-sum-exp
    # soft-min(x) = -logsumexp(-x)/len  approximates min
    # soft-max(x) = logsumexp(x)/len   approximates max
    def soft_min(durations, beta=0.5):
        """Smooth approximation of min using log-sum-exp.
        Higher beta -> sharper (closer to true min)."""
        if not durations:
            return 0.0
        d = np.array(durations, dtype=np.float64)
        if len(d) == 1:
            return float(d[0])
        # soft min = - (1/beta) * log(sum(exp(-beta * d)))
        neg_beta_d = -beta * d
        max_val = np.max(neg_beta_d)
        log_sum = max_val + np.log(np.sum(np.exp(neg_beta_d - max_val)))
        return float(-log_sum / beta)
    
    def soft_max(durations, beta=0.5):
        """Smooth approximation of max using log-sum-exp.
        Higher beta -> sharper (closer to true max)."""
        if not durations:
            return 0.0
        d = np.array(durations, dtype=np.float64)
        if len(d) == 1:
            return float(d[0])
        beta_d = beta * d
        max_val = np.max(beta_d)
        log_sum = max_val + np.log(np.sum(np.exp(beta_d - max_val)))
        return float(log_sum / beta)
    
    min_durations = np.array([
        soft_min(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    max_durations = np.array([
        soft_max(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Average processing time
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Spread: range of machine options (indicates flexibility)
    spread = max_durations - min_durations
    
    # Normalize features
    def normalize(x):
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    nw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    md_norm = normalize(min_durations)  # Higher is longer duration
    ad_norm = normalize(avg_durations)
    mx_norm = normalize(max_durations)
    sp_norm = normalize(spread)
    
    # Time-aware urgency: how much time has elapsed relative to total work
    if time > 0:
        safe_time = max(time, 1e-6)
        urgency = remaining_work / safe_time
        urgency_norm = normalize(urgency)
    else:
        urgency_norm = nw_norm
    
    # Downstream unlock: remaining_ops / (remaining_work + eps)
    # Higher means each unit of work unlocks more future ops
    eps = 1e-6
    unlock_ratio = remaining_ops / (remaining_work + eps)
    unlock_norm = normalize(unlock_ratio)
    
    # Composite score: we want to maximize this
    # Refined hypothesis: 
    # - High remaining work -> critical path (keep pipeline full)
    # - Moderate min duration -> avoid overloading one machine (balance)
    # - High remaining ops -> unlock downstream work early
    # - Time urgency -> don't let jobs starve
    # - Spread -> flexibility to assign to less-loaded machine
    #
    # Key insight: pure LPT (high md_norm) can cause machine imbalance in open-shop.
    # We reduce the weight on min_duration and add a small penalty for very long ops
    # to encourage load balancing across machines.
    
    w_rw = 0.30   # weight for remaining work (critical path)
    w_md = 0.20   # weight for soft-min duration (moderate LPT)
    w_ro = 0.15   # weight for remaining ops (unlock downstream)
    w_urg = 0.15  # weight for time-aware urgency
    w_unlock = 0.10  # weight for downstream unlock ratio
    w_spread = 0.10  # weight for flexibility (spread)
    
    score = (w_rw * nw_norm + 
             w_md * md_norm + 
             w_ro * ro_norm +
             w_urg * urgency_norm +
             w_unlock * unlock_norm +
             w_spread * sp_norm)
    
    # Add a small quadratic penalty for extremely long operations to promote balance
    # This discourages always picking the longest op, which can bottleneck a machine
    if len(min_durations) > 1:
        md_std = np.std(min_durations)
        if md_std > 1e-9:
            md_zscore = (min_durations - min_durations.mean()) / md_std
            # Penalize ops more than 1 std above mean (long ops)
            long_penalty = np.maximum(0, md_zscore - 0.5) ** 2
            long_penalty_norm = normalize(long_penalty)
            score -= 0.05 * long_penalty_norm
    
    # Find the best index
    idx = int(np.argmax(score))
    
    # Decisive tie-breaking: if scores are very close, use a secondary score
    # that prioritizes: (1) more remaining work, (2) shorter min duration (balance),
    # (3) more remaining ops (unlock)
    max_score = score.max()
    close = np.where(score > max_score - 1e-4)[0]
    if len(close) > 1:
        close_rw = remaining_work[close]
        close_md = min_durations[close]
        close_ro = remaining_ops[close]
        
        def normalize_local(x):
            rng = x.max() - x.min()
            if rng > 1e-12:
                return (x - x.min()) / rng
            return np.zeros_like(x)
        
        # Prefer: high remaining work, LOW min duration (to balance), high remaining ops
        sec_score = (0.4 * normalize_local(close_rw) + 
                     0.35 * (1.0 - normalize_local(close_md)) +  # invert: prefer shorter
                     0.25 * normalize_local(close_ro))
        best_local = int(np.argmax(sec_score))
        idx = int(close[best_local])
    
    return int(idx)
# EVOLVE-BLOCK-END
