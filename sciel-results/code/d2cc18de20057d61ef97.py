import numpy as np
import time

def heuristics(demands, capacity):
    # 1. Prepare data
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Keep safe margin before 330s limit
    
    # 2. Compute Compatibility Graph
    # F[i, j] is 1 if item i and j can fit in the same bin
    pair_sums = W[:, None] + W[None, :]
    F = (pair_sums <= capacity).astype(float)
    
    # 3. Compute Structural Metrics
    # Degree of each item: number of items it can fit with
    degrees = F.sum(axis=1)
    
    # Pair Compatibility Score S[i, j]: number of items k that fit with both i and j
    # This is the dot product of row i and row j of F
    # S[i, j] = sum_k F[i, k] * F[j, k]
    S = F @ F
    
    # Gap: capacity - (W[i] + W[j])
    gaps = capacity - pair_sums
    
    # 4. Generate Candidate Heuristics
    # We will try different combinations of structural signals.
    # Base: Fit mask
    fit_mask = F == 1.0
    
    candidates = []
    
    # Variant 1: Pure Inverse Degree (Place hard items early)
    # H[i, j] proportional to 1 / degree[j]
    with np.errstate(divide='ignore', invalid='ignore'):
        h_inv_deg = np.where(fit_mask, 1.0 / (degrees[None, :] + 1.0), 1e-9)
    candidates.append(h_inv_deg)
    
    # Variant 2: Pure Gap Minimization (Standard SRF)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_gap = np.where(fit_mask, 1.0 / (gaps + 1.0), 1e-9)
    candidates.append(h_gap)
    
    # Variant 3: Combined Gap and Inverse Degree
    # H[i, j] = (1/gap) * (1/deg[j])
    with np.errstate(divide='ignore', invalid='ignore'):
        h_combo1 = np.where(fit_mask, (1.0 / (gaps + 1.0)) * (1.0 / (degrees[None, :] + 1.0)), 1e-9)
    candidates.append(h_combo1)
    
    # Variant 4: Combined Gap, Inverse Degree, and Rarity (S)
    # If S[i, j] is low, the pair is "rare" or "tight". 
    # Let's try boosting rare pairs slightly.
    # H[i, j] = (1/gap) * (1/deg[j]) * (1 + 1/S[i, j])
    with np.errstate(divide='ignore', invalid='ignore'):
        rarity_bonus = 1.0 + 1.0 / (S + 1.0)
        h_combo2 = np.where(fit_mask, (1.0 / (gaps + 1.0)) * (1.0 / (degrees[None, :] + 1.0)) * rarity_bonus, 1e-9)
    candidates.append(h_combo2)
    
    # Variant 5: Heavier weight on Rarity
    # H[i, j] = (1/gap) * (1/deg[j]) * (1 + 2.0/S[i, j])
    with np.errstate(divide='ignore', invalid='ignore'):
        rarity_bonus2 = 1.0 + 2.0 / (S + 1.0)
        h_combo3 = np.where(fit_mask, (1.0 / (gaps + 1.0)) * (1.0 / (degrees[None, :] + 1.0)) * rarity_bonus2, 1e-9)
    candidates.append(h_combo3)
    
    # Variant 6: Balance Gap and Degree differently
    # H[i, j] = (1/gap^0.5) * (1/deg[j]^0.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_combo4 = np.where(fit_mask, (1.0 / np.sqrt(gaps + 1.0)) * (1.0 / np.sqrt(degrees[None, :] + 1.0)), 1e-9)
    candidates.append(h_combo4)

    # 5. Evaluate Candidates using ACO Simulation
    eval_seeds = [42, 123, 999, 1234, 5678]
    best_H = None
    best_score = float('inf')
    
    for H in candidates:
        if time.time() > deadline - 20:
            break
            
        # Ensure H is finite and correct shape
        if not np.all(np.isfinite(H)) or H.shape != (n, n):
            continue
            
        # Ensure positive values
        H_clean = np.where(H > 0, H, 1e-9)
        
        # Simulate on a few seeds
        scores = []
        for seed in eval_seeds:
            if time.time() > deadline - 10:
                break
            bins_used = simulate_aco(W, capacity, H_clean, seed=seed)
            scores.append(bins_used)
            
        if len(scores) == len(eval_seeds):
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_H = H_clean
        
    # If no candidates were evaluated or all failed, fall back to a simple heuristic
    if best_H is None:
        best_H = candidates[0] if candidates else np.ones((n, n)) * 1e-9
        
    # Final sanity check
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    
    return best_H

def simulate_aco(demands, capacity, H, ants=20, iters=30, seed=42):
    np.random.seed(seed)
    n = len(demands)
    W = demands
    
    # Pheromone
    tau = np.ones((n, n))
    
    min_bins_global = float('inf')
    
    for it in range(iters):
        bins_counts = []
        for ant in range(ants):
            used = np.zeros(n, dtype=bool)
            bins_used = 0
            items_left = n
            
            while items_left > 0:
                bins_used += 1
                # Find fitting items in empty bin
                unused_indices = np.where(~used)[0]
                if len(unused_indices) == 0:
                    break
                    
                # Random pick for the first item in the bin
                start_idx = unused_indices[np.random.randint(len(unused_indices))]
                used[start_idx] = True
                items_left -= 1
                current = start_idx
                current_load = W[current]
                
                # Now fill bin
                while True:
                    remaining = capacity - current_load
                    fits = (~used) & (W <= remaining)
                    
                    if not np.any(fits):
                        break
                    
                    fit_indices = np.where(fits)[0]
                    
                    # Probabilities proportional to tau[current, j] * H[current, j]
                    p_val = tau[current, fit_indices] * H[current, fit_indices]
                    
                    # Ensure positive
                    p_val = np.where(p_val > 0, p_val, 1e-9)
                    
                    total = np.sum(p_val)
                    if total == 0:
                        next_idx = fit_indices[np.random.randint(len(fit_indices))]
                    else:
                        r = np.random.random() * total
                        cumsum = np.cumsum(p_val)
                        next_local_idx = np.searchsorted(cumsum, r)
                        if next_local_idx >= len(fit_indices):
                            next_local_idx = len(fit_indices) - 1
                        next_idx = fit_indices[next_local_idx]
                        
                    used[next_idx] = True
                    items_left -= 1
                    current_load += W[next_idx]
                    current = next_idx
                    
            bins_counts.append(bins_used)
            
        min_bins_iter = min(bins_counts)
        min_bins_global = min(min_bins_global, min_bins_iter)
        
        # Update Pheromone
        tau = tau * 0.9 + (1.0 / min_bins_iter)
        
    return min_bins_global
