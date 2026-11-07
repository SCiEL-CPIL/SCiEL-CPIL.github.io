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
    
    # Compute degree D[j]: number of items k such that W[j] + W[k] <= capacity
    # D[j] is the count of 1s in the j-th column of fit_mask
    # Since fit_mask is symmetric (W[i]+W[j] == W[j]+W[i]), row sum == col sum
    D = np.sum(fit_mask, axis=0)
    # Avoid division by zero
    D_safe = np.where(D > 0, D, 1)
    inv_D = 1.0 / D_safe
    
    # Precompute some common terms
    inv_gap = 1.0 / (gap + 1e-10)
    inv_gap_15 = 1.0 / (gap + 1e-10)**1.5
    inv_gap_2 = 1.0 / (gap + 1e-10)**2.0
    
    if mode == 'sharp_tight':
        H = np.where(fit_mask, inv_gap, 1e-9)
    elif mode == 'power_sharp':
        H = np.where(fit_mask, inv_gap_15, 1e-9)
    elif mode == 'power_sharp2':
        H = np.where(fit_mask, inv_gap_2, 1e-9)
    elif mode == 'hybrid_sharp':
        H = np.where(fit_mask, W[None, :] * inv_gap, 1e-9)
    elif mode == 'largest_fit':
        H = np.where(fit_mask, W[None, :], 1e-9)
    elif mode == 'tight_fit':
        H = np.where(fit_mask, 1.0 / (gap + 1e-6), 1e-9)
    elif mode == 'scarcity':
        # Prefer items with few compatible partners (large items)
        H = np.where(fit_mask, inv_D[None, :], 1e-9)
    elif mode == 'size_scarcity':
        # Combine size and scarcity
        H = np.where(fit_mask, (W[None, :] * inv_D[None, :]), 1e-9)
    elif mode == 'gap_scarcity':
        # Combine gap and scarcity
        H = np.where(fit_mask, (inv_gap * inv_D[None, :]), 1e-9)
    elif mode == 'size_sq':
        # Strong preference for large items
        H = np.where(fit_mask, W[None, :]**2, 1e-9)
    elif mode == 'scarcity_gap_pow':
        # Strong scarcity signal combined with sharp gap
        H = np.where(fit_mask, (inv_D[None, :]**2) * inv_gap, 1e-9)
    elif mode == 'size_sq_scarcity':
        # Strong size signal combined with scarcity
        H = np.where(fit_mask, (W[None, :]**2) * inv_D[None, :], 1e-9)
    elif mode == 'hybrid_scarcity':
        # Combine size, gap, and scarcity
        H = np.where(fit_mask, (W[None, :] * inv_gap) * inv_D[None, :], 1e-9)
    elif mode == 'blend_gap_scarc':
        # Blend gap and scarcity with equal weight
        H = np.where(fit_mask, (inv_gap + inv_D[None, :]) / 2.0, 1e-9)
    elif mode == 'blend_size_gap':
        # Blend size and gap
        H = np.where(fit_mask, (W[None, :] + inv_gap) / 2.0, 1e-9)
    elif mode == 'blend_triple':
        # Equal blend of size, gap, and scarcity
        H = np.where(fit_mask, (W[None, :] + inv_gap + inv_D[None, :]) / 3.0, 1e-9)
    elif mode == 'gap_sq_scarc':
        # Square gap signal times scarcity
        H = np.where(fit_mask, inv_gap**2 * inv_D[None, :], 1e-9)
    elif mode == 'size_cubed':
        # Strong size preference
        H = np.where(fit_mask, W[None, :]**3, 1e-9)
    elif mode == 'recip_gap_scarc_sq':
        # Reciprocal gap times squared scarcity
        H = np.where(fit_mask, inv_gap * inv_D[None, :]**2, 1e-9)
    elif mode == 'smooth_tight':
        # Smooth tight-fit with moderate exponent
        H = np.where(fit_mask, inv_gap**0.75, 1e-9)
    elif mode == 'adaptive_power':
        # Adaptive: use gap^(-1.25) scaled by scarcity
        H = np.where(fit_mask, inv_gap**1.25 * inv_D[None, :], 1e-9)
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
    # Include new structural modes emphasizing scarcity
    modes = [
        'sharp_tight', 
        'power_sharp', 
        'hybrid_sharp', 
        'power_sharp2', 
        'tight_fit', 
        'largest_fit',
        'scarcity',
        'size_scarcity',
        'gap_scarcity',
        'size_sq',
        'scarcity_gap_pow',
        'size_sq_scarcity',
        'hybrid_scarcity',
        # New blended variants
        'blend_gap_scarc',
        'blend_size_gap',
        'blend_triple',
        'gap_sq_scarc',
        'size_cubed',
        'recip_gap_scarc_sq',
        'smooth_tight',
        'adaptive_power'
    ]

    best_score = float('inf')
    best_H = None

    for mode in modes:
        if time.time() > deadline - 30: # Safety margin for local search
            break

        H = _generate_heuristic(W, capacity, mode)

        try:
            score = _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42)
            if score < best_score:
                best_score = score
                best_H = H
        except Exception:
            continue

    # Local search refinement on best heuristic
    if best_H is not None and time.time() < deadline - 15:
        # Try small multiplicative perturbations on the best heuristic
        for scale in [0.8, 1.2, 0.6, 1.5]:
            if time.time() > deadline - 10:
                break
            H_perturbed = best_H * scale
            H_perturbed = np.where(np.isfinite(H_perturbed) & (H_perturbed > 0), H_perturbed, 1e-9)
            try:
                score = _simulate_aco(demands, capacity, H_perturbed, n_ants=20, n_iterations=30, seed=42)
                if score < best_score:
                    best_score = score
                    best_H = H_perturbed
            except Exception:
                continue

        # Try adding a small scarcity component to the best heuristic
        if time.time() < deadline - 10:
            D = np.sum(np.array(W[:, None] + W[None, :] <= capacity), axis=0)
            D_safe = np.where(D > 0, D, 1)
            inv_D = 1.0 / D_safe
            for alpha in [0.1, 0.3, 0.5]:
                if time.time() > deadline - 5:
                    break
                H_blend = best_H * (1 - alpha) + inv_D[None, :] * alpha
                H_blend = np.where(np.isfinite(H_blend) & (H_blend > 0), H_blend, 1e-9)
                try:
                    score = _simulate_aco(demands, capacity, H_blend, n_ants=20, n_iterations=30, seed=42)
                    if score < best_score:
                        best_score = score
                        best_H = H_blend
                except Exception:
                    continue

    if best_H is None:
        best_H = _generate_heuristic(W, capacity, 'sharp_tight')

    return best_H
