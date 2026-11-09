import numpy as np
import time
import itertools

def simulate_aco(demands, capacity, H, seed, ants=20, iterations=30):
    np.random.seed(seed)
    n = len(demands)
    d = np.array(demands)
    
    # Precompute feasible masks for speed? 
    # H already has 1e-9 for infeasible, but we need to know which are feasible for the 'remaining' logic.
    # Actually, the ACO logic checks `current_load + demands[j] <= capacity`.
    # We can precompute a boolean matrix `feasible[i, j]` to speed up checks if we track last item.
    # However, the 'remaining' set changes.
    
    tau = np.ones((n, n))
    best_overall = float('inf')
    
    # Precompute pair sums for quick feasibility check?
    # pair_sum[i, j] = d[i] + d[j]
    # feasible_pair[i, j] = pair_sum[i, j] <= capacity
    pair_sum = d[:, None] + d[None, :]
    feasible_pair = pair_sum <= capacity
    
    for _ in range(iterations):
        bins_counts = []
        for _ in range(ants):
            remaining = list(range(n))
            bins = 0
            current_load = 0
            last_item = -1
            current_bin_items = []
            
            while remaining:
                if not current_bin_items:
                    # Random fitting item
                    idx = np.random.randint(len(remaining))
                    i = remaining[idx]
                    remaining.pop(idx)
                    current_bin_items.append(i)
                    current_load = d[i]
                    last_item = i
                    bins += 1
                else:
                    # Find feasible j
                    # We need j in remaining such that current_load + d[j] <= capacity
                    # This is equivalent to d[j] <= capacity - current_load
                    max_allowed = capacity - current_load
                    
                    # Filter remaining
                    feasible_indices = []
                    weights = []
                    
                    # Optimization: Instead of looping all remaining, we could use numpy if we kept an array,
                    # but list popping is O(N). N=120 is small.
                    # We iterate through remaining to find feasible ones.
                    for k, j in enumerate(remaining):
                        if d[j] <= max_allowed:
                            feasible_indices.append(k)
                            # Heuristic value
                            # H is precomputed. If H[last_item, j] is 1e-9, it's infeasible or very low.
                            # But we already checked d[j] <= max_allowed, so it's feasible.
                            # The prompt says: "Heuristic entries that are not positive become 1e-9".
                            # Our H construction should ensure positive values for feasible pairs.
                            h_val = H[last_item, j]
                            if h_val <= 0:
                                h_val = 1e-9
                            w = tau[last_item, j] * h_val
                            if w <= 0:
                                w = 1e-9
                            weights.append(w)
                    
                    if not feasible_indices:
                        # Open new bin
                        current_bin_items = []
                        current_load = 0
                        last_item = -1
                        continue
                    
                    # Select j
                    total_w = sum(weights)
                    if total_w == 0:
                        # Fallback to random
                        chosen_k = feasible_indices[np.random.randint(len(feasible_indices))]
                    else:
                        r = np.random.rand() * total_w
                        cumulative = 0
                        chosen_k = feasible_indices[0]
                        for k_idx in range(len(feasible_indices)):
                            cumulative += weights[k_idx]
                            if r <= cumulative:
                                chosen_k = feasible_indices[k_idx]
                                break
                    
                    j = remaining[chosen_k]
                    remaining.pop(chosen_k)
                    current_bin_items.append(j)
                    current_load += d[j]
                    last_item = j
            
            bins_counts.append(bins)
        
        best_this_iter = min(bins_counts)
        if best_this_iter < best_overall:
            best_overall = best_this_iter
        
        # Update Pheromone
        tau *= 0.9
        tau += 1.0 / best_this_iter
        
    return best_overall

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 290 # Safety margin
    
    n = len(demands)
    d = np.array(demands)
    
    # Precompute pair sums and waste
    pair_sum = d[:, None] + d[None, :]
    waste = capacity - pair_sum
    feasible_mask = pair_sum <= capacity
    
    # We will search over alpha and beta for H(i,j) = (pair_sum)^alpha * (waste+1)^beta
    # Note: waste+1 avoids division by zero and ensures positive values.
    # Actually, if waste is 0, (waste+1)=1. If waste is large, it's large.
    # We want to *favor* small waste. So we should use (waste+1)^(-beta) or 1/(waste+1)^beta.
    # Let's use H = (pair_sum)^alpha * (1 / (waste + 1))^beta
    
    # Grid search parameters
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [0.5, 1.0, 1.5, 2.0]
    
    candidates = []
    for a in alphas:
        for b in betas:
            # Calculate H
            # Handle infeasible: set to 1e-9
            # For feasible: (pair_sum)^a * (waste+1)^(-b)
            h = np.zeros((n, n))
            # Avoid 0^0 or similar issues, though pair_sum >= 40 so no 0.
            # waste can be 0. (0+1)^-b = 1.
            h[feasible_mask] = (pair_sum[feasible_mask] ** a) * ((waste[feasible_mask] + 1.0) ** (-b))
            h[~feasible_mask] = 1e-9
            candidates.append(h)
            
    # Also add the original best candidates from the prompt as fallbacks
    # h1: pair_sum
    h1 = np.where(feasible_mask, pair_sum, 1e-9)
    candidates.append(h1)
    # h4: 1/(waste+1)
    h4 = np.where(feasible_mask, 1.0 / (waste + 1.0), 1e-9)
    candidates.append(h4)
    
    # Evaluate candidates
    # We need to be careful with time.
    # 16 grid points + 2 extra = 18 candidates.
    # 3 seeds per candidate.
    # 54 simulations.
    # Each simulation: 30 iter * 20 ants * (120 items * ~60 avg feasible checks) ~ 4.3M ops.
    # Python is slow. 54 * 4.3M is ~230M ops. Might be too slow in 300s.
    # Let's reduce seeds to 2 or grid size.
    # Or, we can estimate time for one simulation.
    
    # Let's time one simulation first to calibrate.
    test_h = candidates[0]
    t0 = time.time()
    simulate_aco(d, capacity, test_h, 42, ants=10, iterations=10) # Smaller run for timing
    t_sim_small = time.time() - t0
    # Scale up: full run is 2x ants, 3x iter -> ~6x time?
    # Actually, ants and iter are linear. 20/10 * 30/10 = 6.
    est_time_full = t_sim_small * 6
    
    # How many full sims can we do in 280s?
    max_sims = int(280 / est_time_full) if est_time_full > 0 else 10
    max_sims = max(1, min(max_sims, 30)) # Cap at 30 to be safe
    
    # We have 18 candidates. If max_sims < 18 * seeds, we need to reduce.
    # Let's use 2 seeds per candidate.
    seeds = [42, 123]
    num_seeds = len(seeds)
    
    # If max_sims < 18 * 2, we might not finish.
    # Let's prioritize the grid search.
    
    best_score = float('inf')
    best_h = candidates[0]
    
    # Sort candidates? Maybe just iterate.
    # We can break early if time runs out.
    
    for idx, h in enumerate(candidates):
        if time.time() > deadline - est_time_full * num_seeds:
            break
            
        scores = []
        for seed in seeds:
            score = simulate_aco(d, capacity, h, seed)
            scores.append(score)
            
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_h = h
            
    return best_h
