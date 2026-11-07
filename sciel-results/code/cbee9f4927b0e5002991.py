"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix combining a data-driven co-occurrence prior 
    from First-Fit-Decreasing (FFD) simulations with a gap-utility reward.
    
    Strategy:
    1. Run multiple randomized FFD simulations to learn which item pairs 
       frequently appear in the same bin in good greedy solutions.
       This provides a structural prior: items that fit well together in 
       practice get higher attractiveness.
    2. For each valid pair (i, j) where d_i + d_j <= capacity, calculate 
       the residual space: residual = capacity - (d_i + d_j).
       Count how many other items k (k != i, k != j) can fit into this 
       residual space.
    3. Combine the two signals:
       - Co-occurrence count C[i, j] rewards structural compatibility.
       - Gap utility rewards tight packing (fewer items fit in the gap).
       The final heuristic is a weighted product or sum that amplifies 
       pairs that are both frequently co-packed AND leave minimal dead space.
       
    This hybrid approach leverages the global structural insight from FFD 
    while maintaining the local efficiency signal from gap analysis.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 20s margin for ACO and overhead
    
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # --- Part 1: Compute gap utility (fast, deterministic) ---
    pair_sums = demands[:, None] + demands[None, :]
    valid_mask = pair_sums <= capacity
    residuals = capacity - pair_sums
    
    # Count how many items fit in each residual (excluding i and j)
    sorted_demands = np.sort(demands)
    counts_all = np.searchsorted(sorted_demands, residuals, side='right')
    
    # Exclude i and j from the count
    fits_i = demands[:, None] <= residuals
    counts_all -= fits_i.astype(int)
    fits_j = demands[None, :] <= residuals
    counts_all -= fits_j.astype(int)
    counts_all = np.maximum(counts_all, 0)
    
    # Gap utility: inversely proportional to (1 + count_k)
    gap_utility = np.full((n, n), 1e-9, dtype=float)
    gap_utility[valid_mask] = 1.0 / (1.0 + counts_all[valid_mask])
    np.fill_diagonal(gap_utility, 1e-9)
    
    # --- Part 2: Compute co-occurrence prior from randomized FFD ---
    C = np.zeros((n, n), dtype=np.int32)
    
    # Pre-sort indices by demand descending
    sorted_indices = np.argsort(-demands)
    
    # Run multiple FFD simulations with randomized tie-breaking
    num_sims = 1000  # Start with a reasonable number
    
    for sim in range(num_sims):
        # Check time
        if time.time() > deadline:
            break
            
        # Create a shuffled version where ties are broken randomly
        noise = np.random.rand(n)
        # Sort by demand descending, then noise for ties
        sort_keys = np.lexsort((noise, -demands))
        
        # FFD Simulation
        bins = []  # List of lists of item indices
        bin_loads = []  # List of current loads
        
        for idx in sort_keys:
            d = demands[idx]
            placed = False
            for b in range(len(bins)):
                if bin_loads[b] + d <= capacity:
                    bins[b].append(idx)
                    bin_loads[b] += d
                    placed = True
                    break
            
            if not placed:
                bins.append([idx])
                bin_loads.append(d)
        
        # Record co-occurrences
        for b in bins:
            if len(b) > 1:
                for i in range(len(b)):
                    for j in range(i + 1, len(b)):
                        u = b[i]
                        v = b[j]
                        C[u, v] += 1
                        C[v, u] += 1
    
    # --- Part 3: Combine gap utility and co-occurrence prior ---
    # Normalize co-occurrence counts to [0, 1]
    max_c = np.max(C)
    if max_c > 0:
        C_norm = C.astype(float) / max_c
    else:
        C_norm = np.zeros((n, n), dtype=float)
    
    # Combine the two signals
    # We want to strongly encourage pairs that are both:
    # 1. Frequently co-occur in good FFD solutions (structural compatibility)
    # 2. Leave minimal dead space (efficiency)
    
    # Option 1: Product - rewards pairs that score well on both dimensions
    # Option 2: Weighted sum - allows one signal to dominate if needed
    # Option 3: Power law on product - amplifies the difference
    
    # Let's use a weighted combination with emphasis on co-occurrence
    # but also require some gap efficiency
    alpha = 0.6  # Weight for co-occurrence
    beta = 0.4   # Weight for gap utility
    
    # Scale gap utility to [0, 1] range
    max_gap = np.max(gap_utility[valid_mask]) if np.any(valid_mask) else 1.0
    if max_gap > 0:
        gap_scaled = gap_utility / max_gap
    else:
        gap_scaled = gap_utility.copy()
    
    # Combine
    eta = np.full((n, n), 1e-9, dtype=float)
    combined = alpha * C_norm + beta * gap_scaled
    eta[valid_mask] = combined[valid_mask]
    
    # Apply a power law to amplify the difference between good and bad pairs
    # This makes the ACO more decisive
    eta[valid_mask] = eta[valid_mask] ** 2.0
    
    # Ensure diagonal is low
    np.fill_diagonal(eta, 1e-9)
    
    # Final sanity check: ensure all values are finite and non-negative
    eta[~np.isfinite(eta)] = 1e-9
    eta = np.maximum(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
