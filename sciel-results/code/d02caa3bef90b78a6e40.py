import numpy as np
import time

def _run_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42):
    """
    Fast ACO simulation. Since pheromone updates are uniform, tau stays constant,
    so selection is purely heuristic-driven. We exploit this for speed.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    eta = np.asarray(heuristic, dtype=np.float64)
    
    # Precompute: for each item i, which items j can follow i (d[i]+d[j] <= capacity)
    # And the eta values for valid transitions
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    
    # For each row i, store valid column indices and their eta values
    valid_cols = [np.where(valid[i])[0] for i in range(n)]
    eta_rows = [eta[i][valid[i]] for i in range(n)]
    
    # Precompute cumulative sums for fast sampling
    eta_cumsums = [np.cumsum(eta_rows[i]) if len(eta_rows[i]) > 0 else np.array([0.0]) for i in range(n)]
    eta_totals = [eta_cumsums[i][-1] if len(eta_rows[i]) > 0 else 0.0 for i in range(n)]
    
    rng = np.random.default_rng(seed)
    best_overall = np.inf
    
    for it in range(n_iters):
        min_bins_this_iter = np.inf
        
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_fill = 0.0
            current_item = -1
            
            # First item: random
            j = int(rng.integers(0, n))
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            while True:
                remaining = capacity - current_fill
                if remaining < 0:
                    break
                
                # Find unused items that fit
                # We need to filter valid_cols[current_item] to only unused items with d[j] <= remaining
                vc = valid_cols[current_item]
                if len(vc) == 0:
                    break
                    
                # Filter: not used and d[j] <= remaining
                mask = (~used[vc]) & (d[vc] <= remaining)
                if not np.any(mask):
                    break
                    
                valid_j = vc[mask]
                if len(valid_j) == 0:
                    break
                    
                # Get eta values for these
                # We need eta[current_item, valid_j]
                eta_vals = eta[current_item, valid_j]
                
                # Sample based on eta
                total = np.sum(eta_vals)
                if total <= 0:
                    # Fallback: uniform
                    j = int(valid_j[int(rng.integers(0, len(valid_j)))])
                else:
                    cum = np.cumsum(eta_vals)
                    cum = cum / cum[-1]
                    r = rng.random()
                    idx = int(np.searchsorted(cum, r))
                    if idx >= len(valid_j):
                        idx = len(valid_j) - 1
                    j = int(valid_j[idx])
                
                used[j] = True
                current_fill += d[j]
                current_item = j
                
                # Check if we're done
                if np.sum(used) == n:
                    break
            else:
                # Check if all items placed
                if not np.all(used):
                    # Open new bin
                    bins += 1
                    # Find first unused item (random among unused)
                    unused_indices = np.where(~used)[0]
                    if len(unused_indices) == 0:
                        break
                    j = int(unused_indices[int(rng.integers(0, len(unused_indices)))])
                    used[j] = True
                    current_fill = d[j]
                    current_item = j
                    continue
                else:
                    break
            
            # After the while loop, check if we need more bins
            if not np.all(used):
                # Count how many more bins needed (rough estimate)
                # Actually, let's just count remaining and estimate
                remaining_count = np.sum(~used)
                if remaining_count > 0:
                    # Each new bin can hold at least 1 item
                    bins += int(np.ceil(remaining_count / 3))  # Rough estimate: ~3 items per bin avg
            
        # Actually, the above loop logic is getting complex. Let me rewrite more carefully.
        # Let me restart the ant simulation with cleaner logic.
        
        # Restart: cleaner ant simulation
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_fill = 0.0
            current_item = -1
            
            # First item: random
            j = int(rng.integers(0, n))
            used[j] = True
            current_fill = d[j]
            current_item = j
            
            placed = 1
            
            while placed < n:
                remaining = capacity - current_fill
                
                # Find unused items that fit in current bin
                vc = valid_cols[current_item]
                if len(vc) > 0:
                    mask = (~used[vc]) & (d[vc] <= remaining)
                    valid_j = vc[mask]
                else:
                    valid_j = np.array([], dtype=int)
                
                if len(valid_j) == 0:
                    # Open new bin
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
                
                # Sample next item from valid_j using eta
                eta_vals = eta[current_item, valid_j]
                total = np.sum(eta_vals)
                if total <= 0:
                    j = int(valid_j[int(rng.integers(0, len(valid_j)))])
                else:
                    cum = np.cumsum(eta_vals)
                    cum = cum / cum[-1]
                    r = rng.random()
                    idx = int(np.searchsorted(cum, r))
                    if idx >= len(valid_j):
                        idx = len(valid_j) - 1
                    j = int(valid_j[idx])
                
                used[j] = True
                current_fill += d[j]
                current_item = j
                placed += 1
            
        # We need to track bins per ant properly. Let me redo this.
        # The issue is I'm not tracking bins correctly in the loop above.
        # Let me rewrite the entire ant loop.
        
        # Clean rewrite
        ant_bins_list = []
        for ant in range(n_ants):
            used = np.zeros(n, dtype=bool)
            bins = 1
            current_fill = 0.0
            current_item = -1
            
            # Place first item randomly
            j = int(rng.integers(0, n))
            used[j] = True
            current_fill = d[j]
            current_item = j
            placed = 1
            
            while placed < n:
                remaining = capacity - current_fill
                
                # Find candidates: unused, fits in remaining space, and valid transition from current_item
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
                
                # Choose next item probabilistically based on eta
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

def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Precompute pair sums and validity
    pair_sum = d[:, None] + d[None, :]
    valid = pair_sum <= capacity
    slack = np.where(valid, capacity - pair_sum, 1e9)
    
    # Complementarity: how well two items fill a bin
    # diff = |d_i + d_j - capacity|; smaller is better (tighter fit)
    # For valid pairs, d_i + d_j <= capacity, so diff = capacity - d_i - d_j = slack
    # But we want to reward tight fits, so 1/(1+slack) rewards small slack
    # For invalid pairs, set to 1e-9
    
    # Base complementarity score
    comp = np.where(valid, 1.0 / (1.0 + slack), 1e-9)
    
    best_score = np.inf
    best_heuristic = None
    
    start_time = time.time()
    time_limit = 280.0  # Leave margin for 3 instances + overhead
    
    # Search over size-weighting exponents and blending with slack-based
    # eta_ij = comp_ij * d_j^gamma * (1/(1+slack_ij))^alpha
    # where alpha controls additional slack preference beyond complementarity
    
    # First, evaluate the pure complementarity score
    base_comp = np.where(np.isfinite(comp) & (comp > 0), comp, 1e-9)
    score = _run_aco_fast(d, capacity, base_comp, n_ants=20, n_iters=30, seed=123)
    best_score = score
    best_heuristic = base_comp.copy()
    
    # Now search over size-weighting gamma
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for gamma in gammas:
        if time.time() - start_time > time_limit:
            break
        if gamma == 0.0:
            candidate = base_comp.copy()
        else:
            bias = np.power(d[None, :], gamma)
            candidate = base_comp * bias
            candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
        
        score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=123)
        if score < best_score:
            best_score = score
            best_heuristic = candidate.copy()
    
    # If time allows, refine around the best gamma
    if time.time() - start_time < time_limit:
        # Find which gamma was best by re-evaluating
        best_gamma = 0.0
        for gamma in gammas:
            if time.time() - start_time > time_limit:
                break
            if gamma == 0.0:
                candidate = base_comp.copy()
            else:
                bias = np.power(d[None, :], gamma)
                candidate = base_comp * bias
                candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
            score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=123)
            if score < best_score:
                best_score = score
                best_heuristic = candidate.copy()
                best_gamma = gamma
        
        # Local search around best_gamma
        for delta in [0.1, -0.1, 0.2, -0.2, 0.3, -0.3]:
            if time.time() - start_time > time_limit:
                break
            g = best_gamma + delta
            if g < 0:
                g = 0.0
            bias = np.power(d[None, :], g)
            candidate = base_comp * bias
            candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
            score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=123)
            if score < best_score:
                best_score = score
                best_heuristic = candidate.copy()
    
    # Also try blending with a pure slack-based heuristic
    if time.time() - start_time < time_limit:
        slack_heur = np.where(valid, 1.0 / (1.0 + slack), 1e-9)
        for blend in [0.25, 0.5, 0.75]:
            if time.time() - start_time > time_limit:
                break
            candidate = (1.0 - blend) * best_heuristic + blend * slack_heur
            candidate = np.where(np.isfinite(candidate) & (candidate > 0), candidate, 1e-9)
            score = _run_aco_fast(d, capacity, candidate, n_ants=20, n_iters=30, seed=123)
            if score < best_score:
                best_score = score
                best_heuristic = candidate.copy()
    
    if best_heuristic is None:
        best_heuristic = np.where(valid, 1.0, 1e-9)
    
    # Final safety: ensure all values are positive and finite
    best_heuristic = np.where(np.isfinite(best_heuristic) & (best_heuristic > 0), best_heuristic, 1e-9)
    
    return best_heuristic
