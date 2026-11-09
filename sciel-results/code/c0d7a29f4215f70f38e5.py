"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # demands is a 1D array of floats
    # Create a 2D array of sums for all pairs (i, j)
    # pair[i, j] = demands[i] + demands[j]
    pair = demands[:, None] + demands[None, :]
    
    # Calculate remaining capacity for each pair
    # rem[i, j] = capacity - (demands[i] + demands[j])
    rem = capacity - pair
    
    # We only care about pairs that fit (rem >= 0)
    # For pairs that don't fit, we set heuristic to 1e-9 (or 0, but task says non-positive becomes 1e-9)
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # So we need to ensure the final matrix has no non-positive values, replacing them with 1e-9.
    
    # Heuristic idea: Favor tighter packs.
    # Inverse of remaining space + 1 to avoid division by zero and scale appropriately.
    # If rem is 0, heuristic is 1/1 = 1.
    # If rem is 100, heuristic is 1/101.
    # This strongly prefers filling the bin.
    
    # Create a mask for valid pairs (rem >= 0)
    valid = rem >= 0
    
    # Initialize heuristic matrix with 1e-9 (invalid/default)
    heur = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Calculate heuristic for valid pairs
    # rem + 1 ensures denominator is at least 1 for integer demands (min rem=1 -> 1/2? No, min rem can be 0 if sum=cap)
    # If sum == capacity, rem=0. Denom = 1. Heur = 1.
    # If sum < capacity, rem > 0.
    # Let's use rem + 1.0 to be safe and smooth.
    heur[valid] = 1.0 / (rem[valid] + 1.0)
    
    # Ensure no non-positive values remain (though 1e-9 is positive)
    # The task says "Heuristic entries that are not positive become 1e-9".
    # Our calculation yields positive values for valid pairs.
    # Invalid pairs are already 1e-9.
    
    return heur
# EVOLVE-BLOCK-END
