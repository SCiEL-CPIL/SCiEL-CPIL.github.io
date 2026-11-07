"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right
    after item i in the same bin.
    
    Strategy: combine the squared utilization (from current program, which
    strongly rewards tight packing) with a complementary-size bonus that
    rewards pairs where the two items are of similar magnitude (both large
    or both small), which tends to produce more uniform bin fills.
    
    The key insight: in bin packing, the best pairs are those that together
    fill the bin closely. The squared term (pair/capacity)^2 already captures
    this. We add a small bonus for pairs where both items are "substantial"
    (not tiny), because tiny items tend to be leftover waste that gets
    scattered across many bins.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Pair sums
    pair = demands[:, None] + demands[None, :]
    
    # Valid mask: pair must fit in the bin
    valid = pair <= capacity
    
    # Base attractiveness: squared utilization, strongly rewards tight fits
    # (pair/capacity)^2 ranges from (40/150)^2≈0.07 to (150/150)^2=1.0
    utilization_sq = np.where(valid, (pair / capacity) ** 2, 0.0)
    
    # Complementary size bonus: reward pairs where both items are reasonably
    # large. The minimum of the two items indicates how "balanced" the pair is.
    # A pair of two 75s is better than 20+130 for fill purposes (same sum,
    # but 20+130 leaves the 20 as potential waste if 130 can't be paired well).
    # Actually, the sum is what matters for fill. Let's use a different angle:
    # reward pairs where the smaller item is still substantial (> 30% of capacity)
    # This avoids pairs like 20+130 where the 20 is likely leftover waste.
    
    # Minimum item size in the pair (normalized)
    min_item = np.minimum(demands[:, None], demands[None, :]) / capacity
    
    # Bonus: smooth ramp that gives credit when min item > ~0.2*capacity
    # This is 0 below 0.15*capacity and ramps to 1 at 0.35*capacity
    min_bonus = np.clip((min_item - 0.15) / 0.20, 0.0, 1.0)
    
    # Combine: base squared utilization + small bonus for balanced pairs
    # Weight the bonus lightly so it doesn't override the primary fill signal
    heuristic = np.where(valid, utilization_sq + 0.05 * min_bonus, 0.0)
    
    # Ensure all entries are positive (minimum 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
