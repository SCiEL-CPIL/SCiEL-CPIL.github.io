import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation exploiting that pheromone is uniform after iter 1.
    After the first iteration, tau is a constant matrix, so selection is
    purely heuristic-driven: P(j|i) ∝ heuristic[i,j].
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    eta = np.asarray(heuristic, dtype=np.float64)
    
    # Precompute valid transitions
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    
    # For each row i, precompute valid column indices
    valid_cols = [np.where(valid[i])[0] for i in range(n)]
    
    rng = np.random.default_rng(seed)
    best_overall = np.inf
    
    for it in range(n_iters):
        ant_bins_list = []
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_fill = 0.0
            current_item = -1
            placed = 0
            
            # Place first item randomly
            j = int(rng.integers(0, n))
            used[j] = True
            current_fill = d[j]
            current_item = j
            placed = 1
            
            while placed < n:
                remaining = capacity - current_fill
                
                # Find candidates: unused, fits, valid transition from current_item
                vc = valid_cols[current_item]
                if len(vc) > 0:
                    mask = (~used[vc]) & (d[vc] <= remaining)
                    valid_j = vc[mask]
                else:
                    valid_j = np.array([], dtype=int)
                
                if len(valid_j) == 0:
                    # No item fits; open new bin
                    bins += 1
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = int(unused_indices[int(rng.integers(0, len(unused_indices)))])
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    placed += 1
                    continue
                
                # Choose next item based on heuristic only (tau is uniform)
                eta_vals = eta[current_item, valid_j]
                total = float(np.sum(eta_vals))
                if total <= 1e-15:
                    j = int(valid_j[int(rng.integers(0, len(valid_j)))])
                else:
                    cum = np.cumsum(eta_vals)
                    cum = cum / cum[-1]
                    r = float(rng.random())
                    idx = int(np.searchsorted(cum, r))
                    if idx >= len(valid_j):
                        idx = len(valid_j) - 1
                    j = int(valid_j[idx])
                
                used[j] = True
                current_fill += d[j]
                current_item = j
                placed += 1
            
            ant_bins_list.append(bins)
        
        min_bins_this_iter = min(ant_bins_list) if ant_bins_list else np.inf
        if min_bins_this_iter < best_overall:
            best_overall = min_bins_this_iter
    
    return best_overall


def _evaluate_robust(d, capacity, candidate, n_seeds=3):
    """Evaluate heuristic with multiple seeds for robustness."""
    scores = []
    for s in [42, 123, 7]:
        score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=s)
        scores.append(score)
    return float(np.mean(scores))


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Precompute pair sums and validity
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    best_score = np.inf
    best_heuristic = None
    
    start_time = time.time()
    time_limit = 290.0  # Leave margin for 3 instances + overhead
    
    # Heuristic family:
    # eta[i,j] = slack_factor^alpha * size_factor^gamma * pair_factor^delta
    # where:
    #   slack_factor = 1/(1+slack)  (prefer tight fits)
    #   size_factor = d[j]          (prefer larger items to fill bins)
    #   pair_factor = (d[i]+d[j])/capacity (prefer pairs that nearly fill capacity)
    
    # Precompute components
    base_slack = np.where(valid, 1.0 / (1.0 + slack), 1e-9)
    base_slack = np.where(np.isfinite(base_slack) & (base_slack > 0), base_slack, 1e-9)
    
    pair_norm = np.where(valid, pair_sum / capacity, 1e-9)
    pair_norm = np.where(np.isfinite(pair_norm) & (pair_norm > 0), pair_norm, 1e-9)
    
    d_col = d[None, :]  # shape (1, n) for broadcasting
    
    # Grid search parameters
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    gammas = [0.0, 0.25, 0.5, 1.0, 1.5]
    deltas = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Precompute powers
    slack_powers = {}
    for alpha in alphas:
        if alpha == 0.0:
            slack_powers[alpha] = np.where(valid, 1.0, 1e-9)
        else:
            val = np.where(valid, np.power(base_slack, alpha), 1e-9)
            slack_powers[alpha] = np.where(np.isfinite(val) & (val > 0), val, 1e-9)
    
    d_powers = {}
    for gamma in gammas:
        if gamma == 0.0:
            d_powers[gamma] = np.ones(n)
        else:
            d_powers[gamma] = np.power(d, gamma)
    
    pair_powers = {}
    for delta in deltas:
        if delta == 0.0:
            pair_powers[delta] = np.where(valid, 1.0, 1e-9)
        else:
            val = np.where(valid, np.power(pair_norm, delta), 1e-9)
            pair_powers[delta] = np.where(np.isfinite(val) & (val > 0), val, 1e-9)
    
    # Evaluate grid
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        for gamma in gammas:
            if time.time() - start_time > time_limit:
                break
            for delta in deltas:
                if time.time() - start_time > time_limit:
                    break
                
                candidate = slack_powers[alpha] * d_powers[gamma][None, :] * pair_powers[delta]
                candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
                
                # Quick single-seed evaluation first
                quick_score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=42)
                
                if quick_score < best_score:
                    # Promising: do robust multi-seed evaluation
                    if time.time() - start_time < time_limit - 10:
                        robust_score = _evaluate_robust(d, capacity, candidate, n_seeds=3)
                        if robust_score < best_score:
                            best_score = robust_score
                            best_heuristic = candidate.copy()
                    else:
                        best_score = quick_score
                        best_heuristic = candidate.copy()
    
    # Local refinement around best parameters
    if time.time() - start_time < time_limit - 30 and best_heuristic is not None:
        # Try to find which (alpha, gamma, delta) gave the best
        # Re-scan to identify best combo
        best_alpha, best_gamma, best_delta = 1.0, 0.5, 1.0
        best_local = np.inf
        
        for alpha in alphas:
            if time.time() - start_time > time_limit - 30:
                break
            for gamma in gammas:
                if time.time() - start_time > time_limit - 30:
                    break
                for delta in deltas:
                    if time.time() - start_time > time_limit - 30:
                        break
                    candidate = slack_powers[alpha] * d_powers[gamma][None, :] * pair_powers[delta]
                    candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
                    score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=42)
                    if score < best_local:
                        best_local = score
                        best_alpha, best_gamma, best_delta = alpha, gamma, delta
        
        # Fine-tune around best
        for da in [-0.25, 0.25, -0.5, 0.5]:
            if time.time() - start_time > time_limit - 20:
                break
            a = best_alpha + da
            if a < 0:
                continue
            val = np.where(valid, np.power(base_slack, a), 1e-9)
            slack_powers[a] = np.where(np.isfinite(val) & (val > 0), val, 1e-9)
            
            for dg in [-0.25, 0.25, -0.5, 0.5]:
                if time.time() - start_time > time_limit - 20:
                    break
                g = best_gamma + dg
                if g < 0:
                    continue
                d_powers[g] = np.power(d, g)
                
                for dd in [-0.25, 0.25, -0.5, 0.5]:
                    if time.time() - start_time > time_limit - 20:
                        break
                    dl = best_delta + dd
                    if dl < 0:
                        continue
                    val = np.where(valid, np.power(pair_norm, dl), 1e-9)
                    pair_powers[dl] = np.where(np.isfinite(val) & (val > 0), val, 1e-9)
                    
                    candidate = slack_powers[a] * d_powers[g][None, :] * pair_powers[dl]
                    candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
                    
                    score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=42)
                    if score < best_score:
                        best_score = score
                        best_heuristic = candidate.copy()
    
    # Try blending with pure best-fit (alpha=1, gamma=0, delta=0)
    if time.time() - start_time < time_limit - 10 and best_heuristic is not None:
        pure_bf = slack_powers.get(1.0, base_slack.copy())
        for blend in [0.2, 0.3, 0.5]:
            if time.time() - start_time > time_limit - 10:
                break
            candidate = (1.0 - blend) * best_heuristic + blend * pure_bf
            candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
            score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=42)
            if score < best_score:
                best_score = score
                best_heuristic = candidate.copy()
    
    # Fallback
    if best_heuristic is None:
        best_heuristic = np.where(valid, 1.0 / (1.0 + slack), 1e-9)
    
    # Final safety
    best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
    
    return best_heuristic
