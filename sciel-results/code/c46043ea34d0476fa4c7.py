import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation using precomputed candidate lists and vectorized operations.
    Returns the minimum number of bins used by any ant.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Precompute: for each item i, which items j can follow i (d[i] + d[j] <= capacity)
    # This is the "valid" mask
    pair_sum = d[:, None] + d[None, :]
    valid_mask = pair_sum <= capacity
    
    # Precompute candidate indices for each item (items that can follow it)
    # For efficiency, we'll use numpy arrays
    candidates_list = [np.where(valid_mask[i])[0] for i in range(n)]
    candidate_counts = np.array([len(c) for c in candidates_list])
    
    # Pheromone initialization
    tau = np.ones((n, n))
    
    best_overall = np.inf
    rng = np.random.default_rng(seed)
    
    for it in range(n_iters):
        ant_bins = []
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_item = -1
            current_fill = 0.0
            
            # Open first bin - place random item
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            # Place remaining items
            while True:
                remaining = capacity - current_fill
                # Get candidates that are unused and fit
                cands = candidates_list[current_item]
                if len(cands) == 0:
                    break
                # Filter to unused items that fit
                valid_cands = cands[~used[cands]]
                if len(valid_cands) == 0:
                    # Open new bin
                    bins += 1
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = unused_indices[rng.integers(0, len(unused_indices))]
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    continue
                
                # Compute probabilities for valid candidates
                row_tau = tau[current_item, valid_cands]
                row_heur = heuristic[current_item, valid_cands]
                probs = row_tau * row_heur
                
                total_prob = np.sum(probs)
                if total_prob <= 0:
                    j = valid_cands[rng.integers(0, len(valid_cands))]
                else:
                    probs = probs / total_prob
                    r = rng.random()
                    # Use searchsorted on cumulative sum
                    cum_probs = np.cumsum(probs)
                    idx = np.searchsorted(cum_probs, r)
                    if idx >= len(valid_cands):
                        idx = len(valid_cands) - 1
                    j = valid_cands[idx]
                
                used[j] = True
                current_fill += d[j]
                current_item = j

            ant_bins.append(bins)
            
        min_bins = min(ant_bins)
        if min_bins < best_overall:
            best_overall = min_bins
            
        # Pheromone update
        tau = tau * 0.9 + (1.0 / min_bins)
        
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 320.0  # Leave margin before 330s limit
    
    # Pre-compute slack matrix
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    # Use the fast ACO runner
    def evaluate_heuristic(base):
        scores = []
        for seed in [42, 123, 456]:
            score = _run_aco_fast(d, capacity, base, n_ants=20, n_iters=30, seed=seed)
            scores.append(score)
        return np.mean(scores)
    
    # Try different heuristic forms
    # Form 1: 1 / (slack + eps)^alpha  (Best Fit style)
    # Form 2: slack^beta (Worst Fit style)
    # Form 3: Combined approaches
    
    # First, try a range of alpha values for best-fit style
    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5]
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        
        if alpha == 0:
            base = np.where(valid, 1.0, 1e-9)
        else:
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            
        # Ensure positive and finite
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        
        avg_score = evaluate_heuristic(base)
        
        if avg_score < best_score:
            best_score = avg_score
            best_heuristic = base.copy()
    
    # Try worst-fit style heuristics
    betas = [0.25, 0.5, 0.75, 1, 1.5, 2]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        base = np.where(valid, (slack + 1e-9)**beta, 1e-9)
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        
        avg_score = evaluate_heuristic(base)
        
        if avg_score < best_score:
            best_score = avg_score
            best_heuristic = base.copy()
    
    # Try combined heuristic: balance between best fit and size complementarity
    # Prefer pairs where d[i] + d[j] is close to capacity but not too close (to leave room)
    if time.time() - start_time < time_limit:
        # Heuristic based on distance from various targets
        for target_frac in [0.5, 0.55, 0.6, 0.65, 0.7]:
            if time.time() - start_time > time_limit:
                break
            target = capacity * target_frac
            dist = np.abs(pair_sum - target)
            base = np.where(valid, 1.0 / (dist + 1e-9), 1e-9)
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            avg_score = evaluate_heuristic(base)
            
            if avg_score < best_score:
                best_score = avg_score
                best_heuristic = base.copy()
    
    # Try hybrid: combine best-fit with worst-fit
    if time.time() - start_time < time_limit:
        for mix in [0.25, 0.5, 0.75]:
            if time.time() - start_time > time_limit:
                break
            best_fit = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
            worst_fit = np.where(valid, (slack + 1e-9)**1.0, 1e-9)
            # Normalize both to [0,1] range for fair mixing
            best_fit_norm = (best_fit - best_fit.min()) / (best_fit.max() - best_fit.min() + 1e-12)
            worst_fit_norm = (worst_fit - worst_fit.min()) / (worst_fit.max() - worst_fit.min() + 1e-12)
            base = np.where(valid, mix * best_fit_norm + (1-mix) * worst_fit_norm, 1e-9)
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            avg_score = evaluate_heuristic(base)
            
            if avg_score < best_score:
                best_score = avg_score
                best_heuristic = base.copy()
    
    # If we have time, do a local refinement around the best alpha found
    if time.time() - start_time < time_limit and best_heuristic is not None:
        # Try finer alpha values around common good values
        for alpha in [0.8, 0.9, 1.1, 1.3, 1.7, 2.5, 3.5]:
            if time.time() - start_time > time_limit:
                break
            
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
            
            avg_score = evaluate_heuristic(base)
            
            if avg_score < best_score:
                best_score = avg_score
                best_heuristic = base.copy()
    
    if best_heuristic is None:
        # Fallback to Best Fit (alpha=1)
        base = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
