"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ant colony optimization
    of offline bin packing.
    
    This version refines the hybrid heuristic by:
    1. Increasing the number of randomized Best-Fit simulations to build a denser
       empirical co-occurrence prior.
    2. Using a smoother, exponential decay function for the static compactness
       score to better distinguish between "tight" and "loose" fits.
    3. Integrating the triple-fit bonus more directly into the static score.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # Safety margin

    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # 1. Static Component Calculation
    pair = d[:, None] + d[None, :]
    fits = pair <= capacity
    
    # Initialize static heuristic
    h_static = np.zeros((n, n), dtype=float)
    
    if not fits.any():
        return np.full((n, n), 1e-9)

    # Gap calculation
    gap = np.where(fits, capacity - pair, 0.0)
    
    # Component 1: Exponential Compactness
    # Rewards small gaps. sigma controls the "steepness".
    # A gap of 0 gets 1.0. A gap of 20 gets exp(-20/sigma).
    # Typical items are 20-99. Capacity 150. Gaps can be up to 130.
    # Let's use sigma = 15.
    sigma = 15.0
    compactness = np.where(fits, np.exp(-gap / sigma), 0.0)
    
    # Component 2: Triple Fit Potential (Vectorized)
    # For each fitting pair, how well does the gap match the next best item?
    sorted_d = np.sort(d)
    
    # Find the largest item that fits in the gap
    # searchsorted returns index where item would be inserted to keep order
    # side='right' means if gap == d[k], it returns k+1. We want k.
    idx = np.searchsorted(sorted_d, gap, side='right') - 1
    idx = np.clip(idx, 0, len(sorted_d) - 1)
    best_fit_size = sorted_d[idx]
    
    # Valid only if the gap is actually >= the smallest item (so idx is meaningful)
    # And if the pair fits.
    min_d = d.min()
    valid_triple = fits & (gap >= min_d)
    
    # Closeness: 1.0 if gap == best_fit_size, 0.0 if gap >> best_fit_size
    # We want to reward if the gap is *close* to an item size.
    # If gap is 50 and best_fit is 49, closeness is high.
    # If gap is 100 and best_fit is 49, closeness is low.
    # Metric: 1 - (gap - best_fit_size) / gap ?
    # Or simply: 1 if gap >= best_fit_size and (gap - best_fit_size) < threshold?
    # Let's use a smooth decay based on the difference.
    diff = np.where(valid_triple, gap - best_fit_size, 0.0)
    # If diff is 0, score 1. If diff is large, score drops.
    # Use exp(-diff / 10)
    triple_score = np.where(valid_triple, np.exp(-diff / 10.0), 0.0)
    
    # Combine static components
    # Compactness is primary. Triple fit is secondary.
    h_static = 0.8 * compactness + 0.2 * triple_score
    
    # Normalize static heuristic to [0, 1]
    max_static = h_static.max()
    if max_static > 0:
        h_static = h_static / max_static

    # 2. Empirical Component Calculation (Co-occurrence Prior)
    # Run lightweight randomized Best-Fit simulations
    co_occurrence = np.zeros((n, n), dtype=float)
    
    # Increase simulations for better statistical reliability
    num_sims = 200
    
    for _ in range(num_sims):
        # Random permutation of items
        perm = list(range(n))
        random.shuffle(perm)
        
        bins = [] # List of lists of item indices
        bin_remaining = [] # List of remaining capacities
        
        for idx in perm:
            item_size = d[idx]
            best_bin_idx = -1
            best_remaining = float('inf')
            
            # Find Best Fit bin
            for b_idx in range(len(bins)):
                if bin_remaining[b_idx] >= item_size:
                    if bin_remaining[b_idx] < best_remaining:
                        best_remaining = bin_remaining[b_idx]
                        best_bin_idx = b_idx
            
            if best_bin_idx == -1:
                # Open new bin
                bins.append([idx])
                bin_remaining.append(capacity - item_size)
            else:
                # Place in best fit bin
                bins[best_bin_idx].append(idx)
                bin_remaining[best_bin_idx] -= item_size
        
        # Update co-occurrence counts for consecutive items in each bin
        for b in bins:
            for k in range(len(b) - 1):
                i = b[k]
                j = b[k+1]
                co_occurrence[i, j] += 1
    
    # Normalize co-occurrence to [0, 1]
    max_co = co_occurrence.max()
    if max_co > 0:
        co_occurrence = co_occurrence / max_co
    
    # 3. Combine Static and Empirical Components
    # Empirical prior is strong, but static provides guidance for unseen pairs.
    # With 200 sims, the empirical matrix is denser.
    alpha = 0.5 # Weight for static
    beta = 0.5  # Weight for empirical
    
    h = alpha * h_static + beta * co_occurrence
    
    # Ensure fitting pairs have a minimum attractiveness
    # Non-fitting pairs remain 0 (will be masked later or set to 1e-9)
    h = np.where(fits, h, 0.0)
    
    # Ensure diagonal is low (don't place item after itself)
    np.fill_diagonal(h, 0.0)
    
    # Convert non-positive/non-finite to 1e-9
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    # Check deadline
    if time.time() > deadline:
        return h

    return h
# EVOLVE-BLOCK-END
