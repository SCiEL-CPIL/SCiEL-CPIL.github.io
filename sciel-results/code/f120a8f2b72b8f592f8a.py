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
                fits_mask = (current_bin_load + W <= capacity) & (~used)
                
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
                
                scores = np.where(fits_mask, scores, 0.0)
                
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
    
    # Precompute degrees (number of compatible partners)
    # deg[j] is count of k such that W[j] + W[k] <= capacity
    # Note: fit_mask[i,j] is True if i and j fit.
    # The degree of j in the compatibility graph is sum(fit_mask[:, j])
    # This includes j itself if 2*W[j] <= capacity.
    deg = np.sum(fit_mask, axis=0)
    # Avoid division by zero, though min weight is 20, cap 150, so 2*20=40<=150, deg >= 1 always?
    # If W[j] > capacity/2, it can only fit with items < capacity/2.
    # Since min W is 20, max is 99. Cap 150.
    # If W[j]=99, need W[k] <= 51. There are items 20..51. So deg > 0.
    # So deg is always >= 1.
    
    hardness = 1.0 / (deg + 1e-10) # Higher hardness for fewer partners
    
    if mode == 'sharp_tight':
        # Pure 1/gap: strongest preference for near-perfect fits
        H = np.where(fit_mask, 1.0 / (gap + 1e-10), 1e-9)
        
    elif mode == 'power_sharp':
        # 1/gap^1.5: even steeper preference
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**1.5, 1e-9)
        
    elif mode == 'power_sharp2':
        # 1/gap^2: steepest preference
        H = np.where(fit_mask, 1.0 / (gap + 1e-10)**2.0, 1e-9)
        
    elif mode == 'hybrid_sharp':
        # w_j / gap: combine size preference with tightness
        H = np.where(fit_mask, W[None, :] / (gap + 1e-10), 1e-9)
        
    elif mode == 'largest_fit':
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    elif mode == 'tight_fit':
        H = np.where(fit_mask, 1.0 / (gap + 1e-6), 1e-9)
        
    elif mode == 'scarcity_tight':
        # 1/gap * hardness(j): prefer tight fits of hard items
        H = np.where(fit_mask, (1.0 / (gap + 1e-10)) * hardness[None, :], 1e-9)
        
    elif mode == 'scarcity_tight_inv':
        # 1/gap * deg(j): prefer tight fits of easy items
        H = np.where(fit_mask, (1.0 / (gap + 1e-10)) * deg[None, :], 1e-9)
        
    elif mode == 'size_scarcity':
        # w_j * hardness(j): prefer large hard items regardless of gap?
        H = np.where(fit_mask, W[None, :] * hardness[None, :], 1e-9)
        
    elif mode == 'gap_scarcity':
        # 1/gap * w_j * hardness(j)
        H = np.where(fit_mask, (1.0 / (gap + 1e-10)) * W[None, :] * hardness[None, :], 1e-9)

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

    # Order candidates by expected quality: sharp tight-fit variants first
    # Added scarcity-based modes
    modes = [
        'sharp_tight', 
        'power_sharp', 
        'hybrid_sharp', 
        'power_sharp2', 
        'tight_fit', 
        'largest_fit',
        'scarcity_tight',
        'gap_scarcity',
        'scarcity_tight_inv',
        'size_scarcity'
    ]

    best_score = float('inf')
    best_H = None

    for mode in modes:
        if time.time() > deadline - 40:
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
