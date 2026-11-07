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
    # Subtract 1 to exclude the item itself
    deg = np.sum(fits, axis=0) - 1.0
    
    # 3. Compute scarcity signal (Structural Centrality)
    # Higher scarcity means fewer partners, so the item is "harder" to place.
    # We use the inverse of the degree + 1 to ensure positivity and scaling.
    # To amplify the difference between scarce and common items, we can use
    # a power function or just the inverse. The directive emphasizes ranking by scarcity.
    scarcity = 1.0 / (deg + 1.0)
    
    # 4. Construct the heuristic matrix
    # The heuristic H[i, j] represents the attractiveness of placing item j
    # after item i. According to the "Most Constrained Variable" heuristic
    # derived from structural centrality, we should prioritize placing items
    # that have the fewest compatible partners (highest scarcity) early in the
    # sequence. Thus, the attractiveness of j is driven primarily by its
    # scarcity.
    
    # Base attractiveness is the scarcity of the destination item j
    H = scarcity[None, :]
    
    # 5. Refine with "Complementarity" based on structural similarity
    # If item i is also scarce, placing another scarce item j after it might
    # be slightly less attractive if we want to diversify the bin, or more
    # attractive if we want to cluster hard-to-place items. 
    # However, the pure "Most Constrained" heuristic suggests that the
    # identity of i matters less than the constraint of j.
    # To add a subtle structural complementarity: if i and j are both very
    # scarce, they are "similar" in their difficulty. We can slightly boost
    # the score if they are both scarce, creating a "hard cluster", or
    # penalize it to encourage mixing. 
    # Given the directive to "rank j by scarcity", we keep the primary driver
    # as scarcity[j]. We will not mix in size or gap metrics.
    
    # 6. Apply mask for valid transitions
    # Only allow transitions where i and j fit and i != j
    valid = fits & (~np.eye(n, dtype=bool))
    H = np.where(valid, H, 1e-9)
    
    # Ensure all values are positive for valid entries
    # The scarcity values are positive, so this is mainly for safety
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
