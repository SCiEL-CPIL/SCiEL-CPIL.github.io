import numpy as np
import time

def _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42):
    """
    Simulate the ACO process to evaluate the heuristic matrix H.
    Returns the mean number of bins used by the best ant in each iteration.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    rng = np.random.default_rng(seed)
    
    # Pheromone initialization
    P = np.ones((n, n))
    
    # Precompute: for each possible current load, which items fit
    # We'll compute fits on the fly but use a faster approach
    # Sort items by weight for potential optimizations
    W_sorted_indices = np.argsort(W)
    
    best_bins_per_iter = []
    
    for it in range(n_iterations):
        min_bins_this_iter = float('inf')
        
        for ant in range(n_ants):
            bins = []
            current_bin_load = 0.0
            current_item = -1
            used = np.zeros(n, dtype=bool)
            
            if n == 0:
                continue
            
            # Place first item randomly
            first_idx = rng.integers(n)
            current_bin_load = W[first_idx]
            current_item = first_idx
            used[first_idx] = True
            
            # Repeatedly pick fitting item
            while True:
                # Vectorized fit check
                remaining_capacity = capacity - current_bin_load
                fits_mask = (W <= remaining_capacity) & (~used)
                
                if not np.any(fits_mask):
                    bins.append(current_bin_load)
                    current_bin_load = 0.0
                    current_item = -1
                    remaining = np.where(~used)[0]
                    if len(remaining) == 0:
                        break
                    first_idx = rng.choice(remaining)
                    current_bin_load = W[first_idx]
                    current_item = first_idx
                    used[first_idx] = True
                    continue
                
                P_row = P[current_item, :]
                H_row = H[current_item, :]
                scores = P_row * H_row
                
                # Zero out non-fitting items
                scores[~fits_mask] = 0.0
                
                total_score = scores.sum()
                if total_score <= 0:
                    candidates = np.where(fits_mask)[0]
                    next_idx = rng.choice(candidates)
                else:
                    probs = scores / total_score
                    next_idx = rng.choice(n, p=probs)
                
                used[next_idx] = True
                current_bin_load += W[next_idx]
                current_item = next_idx
            
            if current_bin_load > 0:
                bins.append(current_bin_load)
            
            if len(bins) < min_bins_this_iter:
                min_bins_this_iter = len(bins)
        
        best_bins_per_iter.append(min_bins_this_iter)
        
        P = P * 0.9 + 1.0 / min_bins_this_iter
    
    return np.mean(best_bins_per_iter)

def _generate_heuristic(W, capacity, mode):
    """
    Generate a heuristic matrix based on the specified mode.
    """
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    gap = capacity - pair_sums
    
    # Compute degree D[j]: number of items k such that W[j] + W[k] <= capacity
    D = np.sum(fit_mask, axis=0)
    D_safe = np.where(D > 0, D, 1)
    inv_D = 1.0 / D_safe
    
    # Remaining capacity after placing both items
    rem_cap = np.maximum(0, capacity - pair_sums)
    
    if mode == 'sharp_tight':
        H = np.where(fit_mask, 1.0 / (gap + 1e-10), 1e-9)
    elif mode == 'power_sharp':
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**1.5, 1e-9)
    elif mode == 'power_sharp2':
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**2.0, 1e-9)
    elif mode == 'power_sharp3':
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**3.0, 1e-9)
    elif mode == 'power_sharp05':
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**0.5, 1e-9)
    elif mode == 'hybrid_sharp':
        H = np.where(fit_mask, W[None, :] / (gap + 1e-10), 1e-9)
    elif mode == 'largest_fit':
        H = np.where(fit_mask, W[None, :], 1e-9)
    elif mode == 'tight_fit':
        H = np.where(fit_mask, 1.0 / (gap + 1e-6), 1e-9)
    elif mode == 'scarcity':
        H = np.where(fit_mask, inv_D[None, :], 1e-9)
    elif mode == 'size_scarcity':
        H = np.where(fit_mask, (W[None, :] * inv_D[None, :]), 1e-9)
    elif mode == 'gap_scarcity':
        H = np.where(fit_mask, (1.0 / (gap + 1e-10)) * inv_D[None, :], 1e-9)
    elif mode == 'size_sq':
        H = np.where(fit_mask, W[None, :]**2, 1e-9)
    elif mode == 'rem_cap':
        # Prefer pairs that leave minimal remaining capacity (tightest fit)
        H = np.where(fit_mask, 1.0 / (rem_cap + 1.0), 1e-9)
    elif mode == 'gap_rem':
        # Combine gap and remaining capacity
        H = np.where(fit_mask, (1.0 / (gap + 1e-10)) * (1.0 / (rem_cap + 1.0)), 1e-9)
    elif mode == 'size_gap':
        # Size weighted by inverse gap
        H = np.where(fit_mask, W[None, :] / (gap + 1e-10), 1e-9)
    elif mode == 'exp_gap':
        # Exponential decay with gap
        H = np.where(fit_mask, np.exp(-gap / 10.0), 1e-9)
    elif mode == 'exp_gap_strong':
        H = np.where(fit_mask, np.exp(-gap / 5.0), 1e-9)
    elif mode == 'log_gap':
        H = np.where(fit_mask, 1.0 / np.log(gap + 1.0 + 1e-10), 1e-9)
    else:
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    """
    start_time = time.time()
    deadline = start_time + 290

    W = np.array(demands, dtype=float)
    n = len(W)

    # Order candidates by expected quality
    modes = [
        'sharp_tight', 
        'power_sharp', 
        'hybrid_sharp', 
        'power_sharp2', 
        'power_sharp3',
        'power_sharp05',
        'tight_fit', 
        'largest_fit',
        'scarcity',
        'size_scarcity',
        'gap_scarcity',
        'size_sq',
        'rem_cap',
        'gap_rem',
        'size_gap',
        'exp_gap',
        'exp_gap_strong',
        'log_gap'
    ]

    best_score = float('inf')
    best_H = None

    for mode in modes:
        if time.time() > deadline - 30: # Safety margin
            break

        H = _generate_heuristic(W, capacity, mode)

        try:
            score = _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42)
            if score < best_score:
                best_score = score
                best_H = H
        except Exception:
            continue

    if best_H is None:
        best_H = _generate_heuristic(W, capacity, 'sharp_tight')

    return best_H
