"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin. Larger is more attractive.
    
    Uses a combination of:
    1. Gap-based fit quality (tighter fits are better)
    2. Size-based FFD bias (larger items preferred)
    3. Co-occurrence prior from randomized First-Fit simulations
    """
    start_time = time.time()
    
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Deadline guard
    def deadline_check():
        return time.time() - start_time > 285
    
    # --- Step 1: Basic gap and size components ---
    
    pair_sum = d[:, None] + d[None, :]
    gap = capacity - pair_sum
    fits = gap >= 0
    np.fill_diagonal(fits, False)
    
    # Gap score: tighter fits have higher score
    gap_score = np.where(fits, 1.0 / (1.0 + gap), 0.0)
    
    # Size bias: favor larger items
    size_bias = d[None, :] / capacity
    
    # Base heuristic
    base_heuristic = gap_score * size_bias
    
    # Deadline guard
    if deadline_check():
        heuristic = np.where(fits, base_heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic
    
    # --- Step 2: Co-occurrence prior from randomized First-Fit simulations ---
    
    # Run multiple randomized First-Fit simulations to build co-occurrence matrix
    n_sims = 50
    co_occurrence = np.zeros((n, n), dtype=np.float64)
    
    indices = np.arange(n)
    
    for sim in range(n_sims):
        if deadline_check():
            break
        
        # Randomly shuffle item order
        order = np.random.permutation(n)
        
        # First-Fit: place each item in the first bin where it fits
        bins = []  # list of (remaining_capacity, [item_indices])
        
        for idx in order:
            item_size = d[idx]
            placed = False
            
            for b in range(len(bins)):
                if bins[b][0] >= item_size:
                    # Place in this bin
                    bins[b] = (bins[b][0] - item_size, bins[b][1] + [idx])
                    placed = True
                    break
            
            if not placed:
                bins.append((capacity - item_size, [idx]))
        
        # Count co-occurrences within each bin
        for bin_items in [b[1] for b in bins]:
            for i_idx in bin_items:
                for j_idx in bin_items:
                    if i_idx != j_idx:
                        co_occurrence[i_idx, j_idx] += 1
    
    # Deadline guard
    if deadline_check():
        # Combine with base heuristic
        # Normalize co-occurrence
        max_co = np.max(co_occurrence) if np.any(co_occurrence > 0) else 1.0
        if max_co > 0:
            co_norm = co_occurrence / max_co
        else:
            co_norm = np.zeros_like(co_occurrence)
        
        # Weighted combination
        alpha = 0.4
        heuristic = (1.0 - alpha) * base_heuristic + alpha * co_norm * base_heuristic
        heuristic = np.where(fits, heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic
    
    # Normalize co-occurrence to [0, 1]
    max_co = np.max(co_occurrence) if np.any(co_occurrence > 0) else 1.0
    if max_co > 0:
        co_norm = co_occurrence / max_co
    else:
        co_norm = np.zeros_like(co_occurrence)
    
    # --- Step 3: Combine all components ---
    
    # The co-occurrence prior captures which items tend to be packed together
    # in good greedy solutions. We use it as a multiplicative boost.
    
    # Final combination: co-occurrence modulates the base heuristic
    # Pairs that co-occur frequently get a boost
    alpha = 0.5  # weight for co-occurrence component
    
    heuristic = (1.0 - alpha) * base_heuristic + alpha * co_norm * base_heuristic
    
    # Ensure non-fitting pairs are 1e-9
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Deadline guard
    if deadline_check():
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
