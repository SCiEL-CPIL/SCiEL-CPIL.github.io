import numpy as np
import time

def _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42):
    """
    Simulate the ACO process to evaluate the heuristic matrix H.
    Returns the mean number of bins used by the best ant in each iteration.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    if n == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    
    # Pheromone initialization
    P = np.ones((n, n))
    
    best_bins_per_iter = []
    
    for it in range(n_iterations):
        min_bins_this_iter = float('inf')
        
        for ant in range(n_ants):
            bins_count = 0
            current_bin_load = 0.0
            current_item = -1
            used = np.zeros(n, dtype=bool)
            
            # Place first item randomly
            first_idx = rng.integers(n)
            current_bin_load = W[first_idx]
            current_item = first_idx
            used[first_idx] = True
            
            # Repeatedly pick fitting item
            while True:
                fits_mask = (current_bin_load + W <= capacity) & (~used)
                
                if not np.any(fits_mask):
                    bins_count += 1
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
                if total_score == 0:
                    candidates = np.where(fits_mask)[0]
                    next_idx = rng.choice(candidates)
                else:
                    probs = scores / total_score
                    next_idx = rng.choice(n, p=probs)
                
                used[next_idx] = True
                current_bin_load += W[next_idx]
                current_item = next_idx
            
            if current_bin_load > 0:
                bins_count += 1
            
            if bins_count < min_bins_this_iter:
                min_bins_this_iter = bins_count
        
        best_bins_per_iter.append(min_bins_this_iter)
        
        P = P * 0.9 + 1.0 / min_bins_this_iter
    
    return np.mean(best_bins_per_iter)


def _compute_scarcity(W, cap):
    """
    Compute scarcity: for each item i, how many other items j can fit with it.
    Items with fewer compatible partners are more scarce and should be placed earlier.
    """
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = (pair_sums <= cap).astype(int)
    # Number of items that can fit with item i (excluding itself)
    compat_counts = fit_mask.sum(axis=1) - 1  # exclude self
    # Scarcity score: higher for fewer compatible partners
    # Normalize: scarcity[i] = 1 / (compat_counts[i] + 1)
    scarcity = 1.0 / (compat_counts + 1.0)
    return scarcity

def _generate_heuristic(W, cap, mode, alpha=1.0, beta=0.0):
    """
    Generate a heuristic matrix based on the specified mode.
    
    alpha: weight of tight-fit (inverse gap) component
    beta: weight of scarcity-based priority component
    """
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= cap
    
    H = np.zeros((n, n))
    
    if mode == 'largest_fit':
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    elif mode == 'tight_fit':
        gap = cap - pair_sums
        H = np.where(fit_mask, 1.0 / (gap + 1e-6), 1e-9)
        
    elif mode == 'hybrid':
        gap = cap - pair_sums
        H = np.where(fit_mask, W[None, :] / (gap + 1.0), 1e-9)
        
    elif mode == 'normalized_tight':
        gap = cap - pair_sums
        H_raw = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        fitting_vals = H_raw[fit_mask]
        if len(fitting_vals) > 0:
            mean_val = np.mean(fitting_vals)
            if mean_val > 0:
                H = H_raw / mean_val
            else:
                H = H_raw
        else:
            H = H_raw
        H = np.where(H > 0, H, 1e-9)
        
    elif mode == 'blend':
        gap = cap - pair_sums
        inv_gap = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        fill_ratio = np.where(fit_mask, pair_sums / cap, 0.0)
        fill_sq = fill_ratio ** 2
        
        inv_gap_fit = inv_gap[fit_mask]
        fill_sq_fit = fill_sq[fit_mask]
        
        if len(inv_gap_fit) > 0:
            mean_inv = np.mean(inv_gap_fit)
            if mean_inv > 0:
                inv_gap = inv_gap / mean_inv
            else:
                inv_gap = np.zeros_like(inv_gap)
                
            mean_fill = np.mean(fill_sq_fit)
            if mean_fill > 0:
                fill_sq = fill_sq / mean_fill
            else:
                fill_sq = np.zeros_like(fill_sq)
        
        H = alpha * inv_gap + (1 - alpha) * fill_sq
        H = np.where(fit_mask, H, 1e-9)
        
    elif mode == 'tight_scarcity':
        """
        Combine tight-fit with scarcity-based priority.
        H[i,j] = alpha * inv_gap[i,j] + beta * scarcity[j] * (W[j] / cap)
        
        The scarcity term encourages placing scarce items (fewer compatible partners) 
        earlier, and the size term encourages larger items.
        """
        gap = cap - pair_sums
        inv_gap = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        
        # Normalize inv_gap to mean 1 for fitting entries
        inv_gap_fit = inv_gap[fit_mask]
        if len(inv_gap_fit) > 0:
            mean_inv = np.mean(inv_gap_fit)
            if mean_inv > 0:
                inv_gap = inv_gap / mean_inv
            else:
                inv_gap = np.zeros_like(inv_gap)
        
        # Scarcity component: for each j, how scarce is it?
        scarcity = _compute_scarcity(W, cap)
        # Size weight: larger items get higher priority
        size_weight = W / cap  # normalize to [~0.13, 0.66]
        # Combine scarcity and size: scarce AND large items get highest priority
        scarcity_size = scarcity * size_weight
        # Normalize scarcity_size to mean 1
        mean_ss = np.mean(scarcity_size)
        if mean_ss > 0:
            scarcity_size = scarcity_size / mean_ss
        
        # H[i,j] = alpha * inv_gap[i,j] + beta * scarcity_size[j]
        # The scarcity_size is a column vector (same for all i in a column)
        H = alpha * inv_gap + beta * scarcity_size[None, :]
        H = np.where(fit_mask, H, 1e-9)
        
    elif mode == 'tight_scarcity_v2':
        """
        Alternative: scarcity affects the tight-fit more directly.
        H[i,j] = inv_gap[i,j] * (1 + beta * scarcity[j])
        """
        gap = cap - pair_sums
        inv_gap = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        
        inv_gap_fit = inv_gap[fit_mask]
        if len(inv_gap_fit) > 0:
            mean_inv = np.mean(inv_gap_fit)
            if mean_inv > 0:
                inv_gap = inv_gap / mean_inv
            else:
                inv_gap = np.zeros_like(inv_gap)
        
        scarcity = _compute_scarcity(W, cap)
        # Boost tight-fit for scarce items
        boost = 1.0 + beta * scarcity[None, :]
        H = inv_gap * boost
        H = np.where(fit_mask, H, 1e-9)
        
    else:
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    # Ensure positivity and finiteness
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H


def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    """
    start_time = time.time()
    deadline = start_time + 290  # 290s limit, keep 10s margin for safety
    
    W = np.array(demands, dtype=float)
    n = len(W)
    
    if n == 0:
        return np.zeros((0, 0))
    
    # Candidate configurations: (mode, alpha, beta)
    # alpha: weight of tight-fit component
    # beta: weight of scarcity/size component
    candidates = []
    
    # Standard modes
    candidates.append(('largest_fit', 1.0, 0.0))
    candidates.append(('tight_fit', 1.0, 0.0))
    candidates.append(('hybrid', 1.0, 0.0))
    candidates.append(('normalized_tight', 1.0, 0.0))
    
    # Blend modes with varying alpha
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    for a in alphas:
        candidates.append(('blend', a, 0.0))
    
    # Tight + scarcity combinations
    # alpha: tight-fit weight, beta: scarcity+size weight
    for a in [0.5, 0.7, 0.9]:
        for b in [0.1, 0.3, 0.5, 0.7]:
            candidates.append(('tight_scarcity', a, b))
    
    # Tight + scarcity v2 (multiplicative boost)
    for b in [0.5, 1.0, 2.0, 3.0]:
        candidates.append(('tight_scarcity_v2', 1.0, b))
    
    best_score = float('inf')
    best_H = None
    
    # Fallback: use a robust default if no time or all fail
    default_H = _generate_heuristic(W, capacity, 'largest_fit')
    
    for mode, alpha, beta in candidates:
        if time.time() > deadline - 30:  # Stop if not enough time left for evaluation
            break
            
        H = _generate_heuristic(W, capacity, mode, alpha, beta)
        
        try:
            score = _simulate_aco(demands, capacity, H, n_ants=20, n_iterations=30, seed=42)
            if score < best_score:
                best_score = score
                best_H = H
        except Exception:
            continue
            
    if best_H is None:
        best_H = default_H
        
    # Final safety check
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    
    return best_H
