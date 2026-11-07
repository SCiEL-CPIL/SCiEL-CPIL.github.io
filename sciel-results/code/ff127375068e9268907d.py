"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    We favor pairs (i, j) that:
    1. Fit within the capacity.
    2. Have a large combined demand (d_i + d_j), as larger items are harder to place.
    3. Have low waste (capacity - d_i - d_j), as tight fits reduce remaining capacity.
    4. Avoid creating 'dead gaps' - remaining space too small for any other item.
       We penalize pairs where the remaining space (capacity - d_i - d_j) is 
       less than the minimum demand of any item, as this space cannot be used.
    
    Formula:
    For fitting pairs:
        remaining = capacity - (d_i + d_j)
        min_demand = min(demands)
        
        If remaining < min_demand (dead gap):
            eta[i,j] = 1e-9  (penalize heavily)
        Else:
            eta[i,j] = (d_i + d_j) / (1.0 + remaining)
    
    For non-fitting pairs: eta[i,j] = 1e-9
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Compute pairwise sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Determine which pairs fit in the bin
    fits = pair_sums <= capacity
    
    # Compute remaining space for fitting pairs
    remaining = capacity - pair_sums
    
    # Minimum demand among all items
    min_demand = np.min(demands)
    
    # Initialize heuristic matrix
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Apply the heuristic for fitting pairs
    if np.any(fits):
        # Identify pairs that fit AND don't create a dead gap
        # A dead gap is when remaining space < min_demand
        valid_pairs = fits & (remaining >= min_demand)
        
        # For valid pairs, compute heuristic
        if np.any(valid_pairs):
            # eta = sum / (1 + remaining)
            # This rewards large sums (large items) and small waste (tight fit)
            # while avoiding dead gaps
            eta[valid_pairs] = pair_sums[valid_pairs] / (1.0 + remaining[valid_pairs])
        
        # For pairs that fit but create a dead gap, keep eta = 1e-9
        # This is already set in the initialization
    
    return eta
# EVOLVE-BLOCK-END
