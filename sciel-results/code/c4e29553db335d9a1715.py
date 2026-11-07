"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.array(demands, dtype=np.float64)
    n = len(demands)
    
    # 1. Compute compatibility matrix (fits)
    # fits[i, j] is True if item i and item j can be in the same bin
    pair_sums = demands[:, None] + demands[None, :]
    fits = pair_sums <= capacity
    
    # 2. Compute degree of each item in the compatibility graph
    # deg[j] is the number of items i (i != j) that fit with j
    diag = np.diag(fits).astype(int)
    deg = np.sum(fits, axis=0) - diag
    
    # 3. Compute scarcity signal (Structural Centrality)
    # Higher scarcity means fewer partners, so the item is "harder" to place
    scarcity = 1.0 / (deg + 1)
    
    # 4. Compute complementarity signal (Fit Tightness)
    # Rewards pairs that fill the bin completely
    gaps = capacity - pair_sums
    tightness = 1.0 / (gaps + 1.0)
    
    # 5. Enhanced structural signal: consider both i's and j's scarcity
    # When placing j after i, we want j to be scarce (hard to place) 
    # but also consider that i itself should have limited options
    # Use geometric mean of scarcities to balance both directions
    combined_scarcity = np.sqrt(scarcity[:, None] * scarcity[None, :])
    
    # 6. Add a size-based complementarity: larger items should pair with smaller
    # to maximize packing efficiency. This is captured by the product of sizes
    # normalized by capacity. Pairs with product close to capacity are best.
    # But we already have tightness which captures this. Let's add a secondary
    # signal: prefer placing larger items earlier (they are harder to fit later)
    size_priority = 1.0 / (demands[None, :] + 1)  # smaller j gets higher priority? 
    # Actually, we want to place LARGE items first because they're harder to place
    # So attractiveness of j should be higher if j is larger
    size_weight = demands[None, :] / capacity
    
    # 7. Combine all signals
    # Primary: tightness (how well they fill the bin)
    # Secondary: scarcity of j (structural constraint)
    # Tertiary: size of j (larger items are harder to place, prioritize them)
    H = tightness * (0.7 * scarcity[None, :] + 0.3 * size_weight)
    
    # 8. Apply mask
    valid = fits & (~np.eye(n, dtype=bool))
    H = np.where(valid, H, 1e-9)
    
    # Ensure all values are positive for valid entries
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
