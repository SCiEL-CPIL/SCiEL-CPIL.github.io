import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    Returns the index of the ready operation to start next.
    
    Strategy: Hybrid MWR + LPT heuristic.
    - Primary: Remaining Work (MWR) to ensure critical path jobs progress.
    - Secondary: Maximum Minimum Duration (LPT-like) to balance loads and prevent late bottlenecks.
    - Tertiary: Remaining Ops count to prioritize job completion.
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0

    start_t = time_module.perf_counter()
    DEADLINE = 145.0

    # Precompute features for all ready operations
    scores = np.zeros(n_ops, dtype=float)
    
    # Store features for tie-breaking and secondary scoring
    rws = np.zeros(n_ops, dtype=float)
    min_durs = np.zeros(n_ops, dtype=float)
    max_durs = np.zeros(n_ops, dtype=float)
    ops_counts = np.zeros(n_ops, dtype=float)
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0.0)
        mo = op.get("machine_options", {})
        
        if not mo:
            scores[i] = -np.inf
            continue
        
        durations = np.array(list(mo.values()), dtype=float)
        min_dur = np.min(durations)
        max_dur = np.max(durations)
        
        # Soft-min surrogate for robustness against noise in single-machine options
        # Using a small temperature to stay close to min but smooth
        # soft_min = -log(sum(exp(-d/t))) * t
        # For small t, approx min.
        # We stick to min_dur for speed, but adjust weights.
        
        rws[i] = rw
        min_durs[i] = min_dur
        max_durs[i] = max_dur
        ops_counts[i] = remaining_ops
        
        # Base score: Remaining Work (MWR)
        # This is the strongest predictor for makespan in open shop.
        score = rw
        
        # LPT Component: Prioritize operations with longer *minimum* durations.
        # Slightly reduced weight to let MWR dominate more.
        score += 0.08 * min_dur
        
        # Flexibility Penalty: Penalize operations with more machine options.
        # Operations with fewer options are more constrained and should be scheduled sooner.
        n_options = len(mo)
        if n_options > 1:
            score -= 0.05 * (n_options - 1)
        
        # Completion Bonus: Jobs with fewer remaining operations are closer to finishing.
        # Finishing a job removes it from the "active" set, potentially simplifying future decisions.
        # This term is small to not override the main heuristics.
        if remaining_ops > 0:
            score += 0.015 / remaining_ops
            
        scores[i] = score
    
    # Check deadline
    if time_module.perf_counter() - start_t > DEADLINE:
        # Fallback: Simple MWR
        return int(np.argmax(rws))
    
    # Find the best index
    best_idx = int(np.argmax(scores))
    
    # Tie-breaking:
    # If scores are very close, we need a robust tie-breaker.
    # Use a larger epsilon to catch near-ties that might be noise.
    max_score = scores[best_idx]
    # Use a relative epsilon for floating point comparison
    epsilon = max(1e-9, 1e-6 * abs(max_score))
    ties = [i for i, s in enumerate(scores) if abs(s - max_score) < epsilon]
    
    if len(ties) > 1:
        # Tie-breaker logic:
        # 1. Max Remaining Work (Most Critical)
        # 2. Fewer Machine Options (More Constrained/Urgent)
        # 3. Min Remaining Ops (Closest to Completion)
        
        # Precompute option counts for tie-breaking
        option_counts = []
        for i in ties:
            mo_i = ready_operations[i].get("machine_options", {})
            option_counts.append(len(mo_i))
        
        def tiebreak_key(idx):
            tie_pos = ties.index(idx)
            return (-rws[idx], option_counts[tie_pos], ops_counts[idx])
        
        best_idx = min(ties, key=tiebreak_key)
        
    return best_idx
