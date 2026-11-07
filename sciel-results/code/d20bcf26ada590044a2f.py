import numpy as np
import time

def heuristics(demands, capacity):
    # 1. Prepare data
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 310  # Keep safe margin before 330s limit
    
    # 2. Compute Compatibility Graph
    # F[i, j] is 1 if item i and j can fit in the same bin
    pair_sums = W[:, None] + W[None, :]
    F = (pair_sums <= capacity).astype(float)
    
    # 3. Compute Structural Metrics
    # Degree of each item: number of items it can fit with
    degrees = F.sum(axis=1)
    
    # Pair Compatibility Score S[i, j]: number of items k that fit with both i and j
    S = F @ F
    
    # Gap: capacity - (W[i] + W[j])
    gaps = capacity - pair_sums
    
    # Size similarity: 1 - |W[i] - W[j]| / capacity
    size_diff = np.abs(W[:, None] - W[None, :])
    size_sim = 1.0 - size_diff / capacity
    
    # 4. Generate Candidate Heuristics
    fit_mask = F == 1.0
    
    candidates = []
    
    # Variant 1: Sharpened Gap (no +1 damping)
    # H[i, j] = 1 / max(gap, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_gap_sharp = np.where(fit_mask, 1.0 / np.maximum(gaps, 1.0), 1e-9)
    candidates.append(h_gap_sharp)
    
    # Variant 2: Sharpened Gap + Inverse Degree
    # H[i, j] = (1 / max(gap,1)) * (1 / (deg[j] + 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        h_gap_deg = np.where(fit_mask, (1.0 / np.maximum(gaps, 1.0)) * (1.0 / (degrees[None, :] + 1.0)), 1e-9)
    candidates.append(h_gap_deg)
    
    # Variant 3: Sharpened Gap + Inverse Degree + Size Similarity
    # H[i, j] = (1 / max(gap,1)) * (1 / (deg[j] + 1)) * (1 + size_sim)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_gap_deg_sim = np.where(fit_mask, (1.0 / np.maximum(gaps, 1.0)) * (1.0 / (degrees[None, :] + 1.0)) * (1.0 + size_sim), 1e-9)
    candidates.append(h_gap_deg_sim)
    
    # Variant 4: Sharpened Gap + Inverse Degree + Rarity
    # H[i, j] = (1 / max(gap,1)) * (1 / (deg[j] + 1)) * (1 + 1/(S+1))
    with np.errstate(divide='ignore', invalid='ignore'):
        rarity_bonus = 1.0 + 1.0 / (S + 1.0)
        h_gap_deg_rare = np.where(fit_mask, (1.0 / np.maximum(gaps, 1.0)) * (1.0 / (degrees[None, :] + 1.0)) * rarity_bonus, 1e-9)
    candidates.append(h_gap_deg_rare)
    
    # Variant 5: Sharpened Gap + Inverse Degree + Size Similarity + Rarity
    # H[i, j] = (1 / max(gap,1)) * (1 / (deg[j] + 1)) * (1 + size_sim) * (1 + 1/(S+1))
    with np.errstate(divide='ignore', invalid='ignore'):
        h_all = np.where(fit_mask, (1.0 / np.maximum(gaps, 1.0)) * (1.0 / (degrees[None, :] + 1.0)) * (1.0 + size_sim) * (1.0 + 1.0 / (S + 1.0)), 1e-9)
    candidates.append(h_all)
    
    # Variant 6: Pure Inverse Degree with sharpened gap weight
    # H[i, j] = (1 / (deg[j] + 1)) * (1 + 1/max(gap,1))
    with np.errstate(divide='ignore', invalid='ignore'):
        h_deg_gap2 = np.where(fit_mask, (1.0 / (degrees[None, :] + 1.0)) * (1.0 + 1.0 / np.maximum(gaps, 1.0)), 1e-9)
    candidates.append(h_deg_gap2)
    
    # Variant 7: Tight-fit complementarity: prioritize placing scarce items into tight fits
    # H[i, j] = (1 / max(gap, 1)) * (1 / (deg[j] + 1)) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        h_tight_scarce = np.where(fit_mask, (1.0 / np.maximum(gaps, 1.0)) * (1.0 / (degrees[None, :] + 1.0)) ** 2, 1e-9)
    candidates.append(h_tight_scarce)
    
    # Variant 8: Complementarity with centrality: items that are central (high degree) 
    # should be placed with items that fit tightly
    # H[i, j] = (1 / max(gap, 1)) * (deg[i] / (deg[j] + 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        h_central = np.where(fit_mask, (1.0 / np.maximum(gaps, 1.0)) * (degrees[:, None] / (degrees[None, :] + 1.0)), 1e-9)
    candidates.append(h_central)
    
    # Variant 9: Gap^1.5 sharpening with inverse degree
    # H[i, j] = (1 / max(gap,1)^1.5) * (1 / (deg[j] + 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        h_gap15_deg = np.where(fit_mask, (1.0 / np.maximum(gaps, 1.0) ** 1.5) * (1.0 / (degrees[None, :] + 1.0)), 1e-9)
    candidates.append(h_gap15_deg)
    
    # Variant 10: Gap^0.5 with inverse degree^1.5
    # H[i, j] = (1 / max(gap,1)^0.5) * (1 / (deg[j] + 1)^1.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_gap05_deg15 = np.where(fit_mask, (1.0 / np.sqrt(np.maximum(gaps, 1.0))) * (1.0 / (degrees[None, :] + 1.0) ** 1.5), 1e-9)
    candidates.append(h_gap05_deg15)
    
    # 5. Evaluate Candidates using ACO Simulation
    eval_seeds = [42, 123, 999, 1234, 5678]
    best_H = None
    best_score = float('inf')
    best_candidate_idx = -1
    best_sim_details = None
    
    for ci, H in enumerate(candidates):
        if time.time() > deadline - 60:
            break
            
        # Ensure H is finite and correct shape
        if not np.all(np.isfinite(H)) or H.shape != (n, n):
            continue
            
        # Ensure positive values
        H_clean = np.where(H > 0, H, 1e-9)
        
        # Simulate on a few seeds
        scores = []
        all_details = []
        for seed in eval_seeds:
            if time.time() > deadline - 50:
                break
            bins_used, details = simulate_aco(W, capacity, H_clean, seed=seed, return_details=True)
            scores.append(bins_used)
            all_details.append(details)
            
        if len(scores) == len(eval_seeds):
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_H = H_clean
                best_candidate_idx = ci
                best_sim_details = all_details
        
    # If no candidates were evaluated or all failed, fall back to a simple heuristic
    if best_H is None:
        best_H = candidates[0] if candidates else np.ones((n, n)) * 1e-9
        best_sim_details = None
    
    # 6. Refinement pass: adjust heuristic based on simulation outcomes
    if best_sim_details is not None and time.time() < deadline - 10:
        best_H = refine_heuristic(best_H, W, capacity, best_sim_details, fit_mask, deadline)
    
    # Final sanity check
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    
    return best_H

def refine_heuristic(H_base, W, capacity, sim_details, fit_mask, deadline):
    """
    Refine the heuristic by analyzing which (i,j) transitions led to good solutions.
    For each simulation, track which pairs were used in the best-performing ant's solution.
    Boost pairs that appear in tight, successful placements; dampen others.
    """
    n = len(W)
    # Aggregate: count how often each (i,j) pair was used in good solutions
    # "Good" = the ant that used the fewest bins in that iteration
    pair_usage = np.zeros((n, n))
    pair_tightness = np.zeros((n, n))
    
    for details in sim_details:
        if not details:
            continue
        for ant_data in details:
            if ant_data is None:
                continue
            # ant_data: list of (current, next) pairs that were selected
            for (i, j, tightness) in ant_data:
                pair_usage[i, j] += 1
                pair_tightness[i, j] += tightness
    
    # Only refine entries that are in the fit mask
    # Boost: H_refined[i,j] = H_base[i,j] * (1 + alpha * usage[i,j])
    # where usage is normalized
    max_usage = np.max(pair_usage) if np.max(pair_usage) > 0 else 1.0
    usage_norm = pair_usage / max_usage
    
    # Alpha controls strength of refinement
    alpha = 0.3
    H_refined = np.where(fit_mask, H_base * (1.0 + alpha * usage_norm), 1e-9)
    
    # Ensure positivity and finiteness
    H_refined = np.where(np.isfinite(H_refined) & (H_refined > 0), H_refined, 1e-9)
    
    # Quick validation: run one more simulation to ensure refinement helps
    if time.time() < deadline - 5:
        test_score, _ = simulate_aco(W, capacity, H_refined, seed=42, return_details=False)
        base_score, _ = simulate_aco(W, capacity, np.where(H_base > 0, H_base, 1e-9), seed=42, return_details=False)
        if test_score > base_score:
            # Refinement hurt; fall back to base
            H_refined = np.where(np.isfinite(H_base) & (H_base > 0), H_base, 1e-9)
    
    return H_refined

def simulate_aco(demands, capacity, H, ants=20, iters=30, seed=42, return_details=False):
    np.random.seed(seed)
    n = len(demands)
    W = demands
    
    # Pheromone
    tau = np.ones((n, n))
    
    min_bins_global = float('inf')
    all_details = []
    
    for it in range(iters):
        bins_counts = []
        ant_details_list = []
        for ant in range(ants):
            used = np.zeros(n, dtype=bool)
            bins_used = 0
            items_left = n
            ant_pairs = []
            
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
                        
                    # Track pair for refinement
                    tightness = 1.0 / max(capacity - current_load - W[next_idx], 1.0)
                    ant_pairs.append((current, next_idx, tightness))
                    
                    used[next_idx] = True
                    items_left -= 1
                    current_load += W[next_idx]
                    current = next_idx
                    
            bins_counts.append(bins_used)
            ant_details_list.append(ant_pairs)
            
        min_bins_iter = min(bins_counts)
        min_bins_global = min(min_bins_global, min_bins_iter)
        
        # Store details for the best ant in this iteration
        best_ant_idx = bins_counts.index(min_bins_iter)
        all_details.append(ant_details_list[best_ant_idx])
        
        # Update Pheromone
        tau = tau * 0.9 + (1.0 / min_bins_iter)
        
    if return_details:
        return min_bins_global, all_details
    return min_bins_global
