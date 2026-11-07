"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair_sum = demands[:, None] + demands[None, :]
    gap = capacity - pair_sum
    
    # Initialize with very small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    # Valid pairs are those where both items fit together
    valid = gap >= 0
    
    # For each valid pair (i, j), compute how many other items could fit
    # into the remaining space gap[i,j] = capacity - d_i - d_j
    # This rewards pairs that leave "dead gaps" (too small for any remaining item)
    # or exact fits, rather than large unused voids.
    
    # Build a cumulative count of items by demand size
    demand_ints = np.round(demands).astype(int)
    
    max_demand = int(np.max(demand_ints))
    
    # Create a frequency array
    freq = np.zeros(max_demand + 1, dtype=np.int64)
    for d in demand_ints:
        freq[d] += 1
    
    # Cumulative sum: cumsum[k] = number of items with demand <= k
    cumsum = np.cumsum(freq)
    
    # Get indices of valid pairs
    valid_indices = np.where(valid)
    i_idx = valid_indices[0]
    j_idx = valid_indices[1]
    
    gap_valid = gap[valid]
    
    # Clip gap to valid range for cumsum lookup
    gap_clipped = np.clip(gap_valid, 0, max_demand).astype(int)
    
    # Count of items with demand <= gap
    fit_counts = cumsum[gap_clipped]
    
    # Check if item i fits in gap[i,j]: 2*d_i + d_j <= capacity
    i_fits = (2.0 * demands[i_idx] + demands[j_idx]) <= capacity
    # Check if item j fits in gap[i,j]: d_i + 2*d_j <= capacity
    j_fits = (demands[i_idx] + 2.0 * demands[j_idx]) <= capacity
    
    # Adjust fit_counts: subtract 1 for each item that fits and is i or j
    adjusted_counts = fit_counts - i_fits.astype(np.int64) - j_fits.astype(np.int64)
    
    # Ensure counts are non-negative
    adjusted_counts = np.maximum(adjusted_counts, 0)
    
    # The heuristic should reward pairs with FEWER fitting items in the residual
    # Use a smooth penalty: pairs with 0 fitting items get highest score
    H_valid = 1.0 / (adjusted_counts + 1.0)
    
    # Add a bonus for smaller gaps (tighter packing)
    # Normalize gap to [0,1] range where 0 is tightest
    gap_normalized = gap_valid / capacity
    # Reward tighter packs: multiply by (1 - gap_normalized) to boost small gaps
    H_valid = H_valid * (1.0 - 0.5 * gap_normalized)
    
    H[valid] = H_valid
    
    # Boost exact fits even more
    exact_fit = valid & (gap == 0)
    H[exact_fit] = 15.0
    
    return H
# EVOLVE-BLOCK-END
