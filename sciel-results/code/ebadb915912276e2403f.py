"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = demands.astype(np.float64)
    
    # Pair sums: demands[i] + demands[j]
    pair_sum = d[:, None] + d[None, :]
    
    # Waste: how much space is left after placing both items
    waste = capacity - pair_sum
    
    # Only valid pairs where both items fit together
    valid = pair_sum <= capacity
    
    # Initialize with a very small value for invalid pairs
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    # For valid pairs, compute a strong non-linear reward for tight fits
    # Using 1/waste^2 gives a very strong preference for pairs that nearly fill the bin
    # Add a small baseline to avoid division by zero (waste can be 0 if pair_sum == capacity)
    safe_waste = np.where(valid, waste, 1.0)
    
    # Squared inverse waste: strongly rewards tight fits
    # When waste is small (tight fit), this value is large
    # When waste is large (loose fit), this value is small
    inv_waste_sq = 1.0 / (safe_waste ** 2)
    
    # Scale to reasonable range
    # Maximum possible waste is capacity - 2*min_demand, minimum is 0
    # inv_waste_sq can be very large when waste is close to 0
    # Let's cap it to avoid numerical issues
    max_inv = np.max(inv_waste_sq[valid]) if np.any(valid) else 1.0
    if max_inv > 0:
        inv_waste_sq = inv_waste_sq / max_inv
    
    # Also add a component that rewards larger items (they're harder to place)
    # This helps the ACO find good placements for large items
    size_bonus = (d[:, None] + d[None, :]) / capacity
    
    # Combine: primary signal is tight fit, secondary is size
    H[valid] = inv_waste_sq[valid] + 0.1 * size_bonus[valid]
    
    # Ensure all values are positive and finite
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
# EVOLVE-BLOCK-END
