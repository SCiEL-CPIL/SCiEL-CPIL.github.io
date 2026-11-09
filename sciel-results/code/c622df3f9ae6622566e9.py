"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    Picks the ready operation to start next.
    
    Strategy: Combine multiple heuristics:
    1. Remaining work (total) - prioritize jobs with more work
    2. Minimum processing time on any machine - prioritize short ops to reduce makespan
    3. Number of remaining operations - prioritize jobs with more ops (to unblock them)
    4. Machine load balancing - avoid overloading machines
    
    We use a weighted combination that adapts based on the situation.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Minimum duration across all machine options for each op
    min_durs = np.zeros(n, dtype=np.float64)
    max_durs = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if opts:
            vals = list(opts.values())
            min_durs[i] = min(vals)
            max_durs[i] = max(vals)
    
    # Normalize features to [0, 1] range for weighted combination
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    nw = normalize(remaining_works)
    no = normalize(remaining_ops)
    nd = normalize(min_durs)
    
    # Invert min duration: we want to prioritize SHORTER operations (to reduce makespan)
    # But also consider that longer remaining work means the job is more critical
    # Use a combined score
    
    # Heuristic 1: SPT (Shortest Processing Time) - prioritize short ops
    # This tends to minimize makespan in open shop
    
    # Heuristic 2: Most work remaining - prioritize jobs with more total work
    # This can help complete jobs faster
    
    # Heuristic 3: Most remaining ops - prioritize jobs that have more ops to go
    # This helps unblock machines sooner
    
    # Weighted combination:
    # - Strong weight on short processing time (SPT is known to be good for makespan)
    # - Moderate weight on remaining work
    # - Small weight on remaining ops
    
    # Dynamic weighting: early on, use more SPT; later, use more remaining work
    # We don't have direct access to total makespan, but we can use 'time' as a proxy
    # Early: time is small relative to total work
    total_work = remaining_works.sum()
    if total_work > 0:
        progress = min(time / (total_work * 2), 1.0)  # rough progress estimate
    else:
        progress = 1.0
    
    # Weights that shift from SPT-heavy to work-heavy as we progress
    w_spt = 1.0 - 0.5 * progress      # 1.0 early, 0.5 late
    w_work = 0.5 + 0.5 * progress     # 0.5 early, 1.0 late
    w_ops = 0.2                        # constant small weight
    
    # Score: higher is better
    # For SPT: shorter is better, so use (1 - nd)
    # For work: more is better, so use nw
    # For ops: more is better, so use no
    
    scores = w_spt * (1.0 - nd) + w_work * nw + w_ops * no
    
    # Tie-breaking: if scores are very close, prefer the one with shorter min duration
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
