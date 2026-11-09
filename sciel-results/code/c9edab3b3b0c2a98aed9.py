import numpy as np
import time

def _run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Simulates the ACO process and returns the minimum number of bins used by any ant.
    """
    n = len(demands)
    d = np.array(demands, dtype=float)
    
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
            
            # Open first bin
            j = rng.integers(0, n)
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            # Place remaining items
            while True:
                remaining = capacity - current_fill
                mask = (~used) & (d <= remaining)
                
                if not np.any(mask):
                    bins += 1
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = unused_indices[rng.integers(0, len(unused_indices))]
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    continue
                else:
                    row_tau = tau[current_item, :]
                    row_heur = heuristic[current_item, :]
                    probs = row_tau * row_heur
                    probs[~mask] = 0.0
                    
                    total_prob = np.sum(probs)
                    if total_prob <= 0:
                        candidates = np.where(mask)[0]
                        j = candidates[rng.integers(0, len(candidates))]
                    else:
                        probs = probs / total_prob
                        cum_probs = np.cumsum(probs)
                        r = rng.random()
                        idx = np.searchsorted(cum_probs, r)
                        if idx >= n:
                            idx = n - 1
                        j = idx
                        
                    used[j] = True
                    current_fill += d[j]
                    current_item = j

            ant_bins.append(bins)
            
        min_bins = min(ant_bins)
        if min_bins < best_overall:
            best_overall = min_bins
            
        tau = tau * 0.9 + (1.0 / min_bins)
        
    return best_overall

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    start_time = time.time()
    time_limit = 310.0  # Leave margin before 330s limit
    
    # Pre-compute slack matrix
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    def _eval_heuristic(base):
        """Evaluate a heuristic matrix with 3 seeds, return mean score."""
        scores = []
        for seed in [42, 123, 456]:
            score = _run_aco(d, capacity, base, n_ants=20, n_iters=30, seed=seed)
            scores.append(score)
        return np.mean(scores)
    
    def _try_heuristic(base):
        """Try a heuristic, update best if better."""
        nonlocal best_score, best_heuristic
        base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        avg_score = _eval_heuristic(base)
        if avg_score < best_score:
            best_score = avg_score
            best_heuristic = base.copy()
        return avg_score
    
    # Phase 1: Coarse search over best-fit alpha values
    alphas = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10]
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        
        if alpha == 0:
            base = np.where(valid, 1.0, 1e-9)
        else:
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
        
        _try_heuristic(base)
    
    # Phase 2: Worst-fit style
    betas = [0.5, 1, 1.5, 2, 3]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        base = np.where(valid, (slack + 1e-9)**beta, 1e-9)
        _try_heuristic(base)
    
    # Phase 3: Target-distance heuristics (prefer pairs summing near a target)
    targets = [capacity * 0.5, capacity * 0.55, capacity * 0.6, capacity * 0.65, capacity * 0.7, capacity * 0.8]
    
    for target in targets:
        if time.time() - start_time > time_limit:
            break
        
        dist = np.abs(pair_sum - target)
        base = np.where(valid, 1.0 / (dist + 1.0), 1e-9)
        _try_heuristic(base)
    
    # Phase 4: Complementarity heuristic - reward pairs that fill bins well
    # Use exponential of negative slack (strongly prefer tight fits)
    for gamma in [0.05, 0.1, 0.15, 0.2, 0.3]:
        if time.time() - start_time > time_limit:
            break
        
        base = np.where(valid, np.exp(-gamma * slack), 1e-9)
        _try_heuristic(base)
    
    # Phase 5: Combined - product of best-fit and complementarity
    if time.time() - start_time < time_limit:
        for alpha in [1, 2, 3]:
            for gamma in [0.1, 0.15, 0.2]:
                if time.time() - start_time > time_limit:
                    break
                bf = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
                comp = np.where(valid, np.exp(-gamma * slack), 1e-9)
                base = bf * comp
                _try_heuristic(base)
    
    # Phase 6: Fine-grained search around best alpha if we have time
    if time.time() - start_time < time_limit * 0.7:
        # Fine grid for best-fit
        fine_alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
        for alpha in fine_alphas:
            if time.time() - start_time > time_limit:
                break
            base = np.where(valid, 1.0 / (slack + 1e-9)**alpha, 1e-9)
            _try_heuristic(base)
    
    # Phase 7: Size-based heuristic - prefer placing larger items with smaller items
    if time.time() - start_time < time_limit:
        # Heuristic: 1 / (d[i] * d[j]) for valid pairs - favors complementary sizes
        size_prod = d[:, None] * d[None, :]
        base = np.where(valid, 1.0 / (size_prod + 1e-9), 1e-9)
        _try_heuristic(base)
        
        # Heuristic: (capacity - d[i] - d[j])^(-1) * d[i] * d[j]
        # This rewards pairs that are both large and fit well
        base2 = np.where(valid, (d[:, None] * d[None, :]) / (slack + 1.0), 1e-9)
        _try_heuristic(base2)
    
    if best_heuristic is None:
        base = np.where(valid, 1.0 / (slack + 1e-9), 1e-9)
        best_heuristic = np.where(np.isfinite(base) & (base > 0), base, 1e-9)
    
    return best_heuristic
