import numpy as np
import time

def simulate_aco_vectorized(demands, capacity, H, seed, n_iterations=30, n_ants=20):
    """
    Vectorized ACO simulation to quickly evaluate a heuristic.
    Returns the minimum number of bins used by any ant across all iterations.
    """
    np.random.seed(seed)
    n = len(demands)
    d = np.array(demands)
    
    tau = np.ones((n, n))
    best_overall = float('inf')
    
    for _ in range(n_iterations):
        bins_counts = []
        
        for _ in range(n_ants):
            remaining_indices = np.arange(n)
            bins = 0
            current_load = 0.0
            last_item = -1
            
            while len(remaining_indices) > 0:
                if last_item == -1:
                    # Open new bin: pick random fitting item
                    # Since all items fit individually (max 99 < 150), any item fits.
                    idx = np.random.randint(len(remaining_indices))
                    item = remaining_indices[idx]
                    remaining_indices = np.delete(remaining_indices, idx)
                    current_load = d[item]
                    last_item = item
                    bins += 1
                else:
                    # Find feasible items
                    # We need j such that current_load + d[j] <= capacity
                    # And we use probability proportional to tau[last_item, j] * H[last_item, j]
                    
                    # Vectorized feasibility check
                    # remaining_indices are indices into the original array
                    loads_of_remaining = d[remaining_indices]
                    feasible_mask = (current_load + loads_of_remaining) <= capacity
                    
                    if not np.any(feasible_mask):
                        # Open new bin
                        last_item = -1
                        current_load = 0.0
                        continue
                    
                    # Get weights
                    # H and tau are indexed by original indices
                    # We need to map remaining_indices to their corresponding row/col in H/tau
                    # tau[last_item, remaining_indices] and H[last_item, remaining_indices]
                    
                    w = tau[last_item, remaining_indices] * H[last_item, remaining_indices]
                    
                    # Mask out infeasible
                    w[~feasible_mask] = 0.0
                    
                    # Handle non-positive weights (shouldn't happen if H is positive, but safe)
                    # The task says "Heuristic entries that are not positive become 1e-9"
                    # So H should be positive. But let's be safe.
                    w = np.maximum(w, 1e-9)
                    w[~feasible_mask] = 0.0 # Re-apply mask after max, or do it properly
                    
                    # Actually, if H is positive, w is positive for feasible.
                    # Let's just normalize.
                    total_w = np.sum(w)
                    if total_w == 0:
                        # Fallback to random if something went wrong (should not happen)
                        idx = np.random.randint(len(remaining_indices))
                    else:
                        probs = w / total_w
                        # np.random.choice with probabilities
                        # We need to pick from the range(len(remaining_indices))
                        idx = np.random.choice(len(remaining_indices), p=probs)
                    
                    item = remaining_indices[idx]
                    remaining_indices = np.delete(remaining_indices, idx)
                    current_load += d[item]
                    last_item = item
            
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
    deadline = start_time + 295  # Safety margin for 300s limit
    
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Precompute pair sums
    pair_sum = d[:, None] + d[None, :]
    feasible_mask = pair_sum <= capacity
    
    # We want to search for a heuristic function H[i, j].
    # The ACO rule prefers higher H.
    # Usually, "Best Fit" (min waste) or "Worst Fit" (max waste) are good heuristics for bin packing.
    # In ACO, if we want to prefer tight fits (to save space for larger items later or just fill bins),
    # we want H to be high when waste is low.
    # So H ~ 1/waste or exp(-waste).
    # Or if we want to pack large items together?
    
    # Let's generate a diverse set of candidates.
    candidates = []
    
    # 1. Inverse Waste (Best Fit)
    waste = capacity - pair_sum
    # Avoid division by zero. If waste=0, it's a perfect fit, very good.
    # 1/(waste + eps)
    for eps in [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        h = np.where(feasible_mask, 1.0 / (waste + eps), 1e-9)
        candidates.append(h)
        
    # 2. Exponential Decay of Waste
    for scale in [5.0, 10.0, 20.0, 50.0, 100.0]:
        h = np.where(feasible_mask, np.exp(-waste / scale), 1e-9)
        candidates.append(h)
        
    # 3. Ratio based?
    # Maybe favor placing large items after large items?
    # H = d[i] * d[j]?
    prod = d[:, None] * d[None, :]
    h_prod = np.where(feasible_mask, prod, 1e-9)
    candidates.append(h_prod)
    
    # 4. Sum based?
    h_sum = np.where(feasible_mask, pair_sum, 1e-9)
    candidates.append(h_sum)
    
    # 5. Power of Inverse Waste
    for p in [0.5, 1.0, 2.0, 3.0]:
        h = np.where(feasible_mask, (1.0 / (waste + 1.0)) ** p, 1e-9)
        candidates.append(h)
        
    # 6. Hybrid: Inverse Waste + Small Constant
    # This avoids extreme values
    
    best_score = float('inf')
    best_h = candidates[0]
    
    # Use a few seeds for robust selection
    seeds = [42, 123, 7, 99, 2024]
    
    # Evaluate candidates
    # We need to be careful with time.
    # simulate_aco_vectorized is fast, but let's check.
    # 30 iters * 20 ants * 120 items ~ 72000 steps.
    # Vectorized numpy ops are fast.
    
    num_candidates = len(candidates)
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 5:
            break
            
        # Evaluate on a subset of seeds first?
        # Let's just do all seeds, but break if time is tight.
        
        scores = []
        for seed in seeds:
            if time.time() > deadline - 2:
                break
            score = simulate_aco_vectorized(d, capacity, h, seed)
            scores.append(score)
            
        if not scores:
            continue
            
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_h = h
            
        # Optional: If we find a very good score, we might stop?
        # Lower bound is roughly sum(d)/capacity.
        # sum(d) ~ 120 * 60 = 7200. 7200/150 = 48.
        # If we get 49, it's very good.
        if avg_score <= 49:
            # Could stop, but let's see if others are better.
            # For now, continue to find the absolute best among candidates.
            pass

    return best_h
