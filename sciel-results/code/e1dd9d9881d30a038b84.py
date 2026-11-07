"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin, using a complementarity-score approach.
    
    Uses a two-pointer sweep on sorted demands to identify ideal complementary
    pairs, then assigns high attractiveness to pairs that are near-optimal
    complements while penalizing large deviations from ideal pairing.
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Sort demands to identify complementarity structure
    sorted_indices = np.argsort(demands)
    sorted_vals = demands[sorted_indices]
    
    # Two-pointer: for each position i in sorted array, find the largest
    # position j such that sorted_vals[i] + sorted_vals[j] <= capacity
    # This gives the ideal complement for each item
    best_j = np.zeros(n, dtype=np.int64)
    best_sum = np.zeros(n, dtype=np.float64)
    
    j_ptr = n - 1
    for i in range(n):
        while j_ptr >= i and sorted_vals[i] + sorted_vals[j_ptr] > capacity:
            j_ptr -= 1
        if j_ptr >= i:
            best_j[i] = j_ptr
            best_sum[i] = sorted_vals[i] + sorted_vals[j_ptr]
        else:
            best_j[i] = i  # fallback: pair with self if nothing else fits
            best_sum[i] = 2 * sorted_vals[i] if 2 * sorted_vals[i] <= capacity else 0.0
    
    # Map sorted positions back to original indices
    sort_pos = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        sort_pos[sorted_indices[idx]] = idx
    
    # For each original item, get its ideal complement sum
    ideal_sum = best_sum[sort_pos]
    
    # Compute pairwise sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Valid pairs: sum <= capacity
    valid = pair_sums <= capacity
    
    # For valid pairs, compute fill ratio (how close to capacity)
    fill_ratio = np.where(valid, pair_sums / capacity, 0.0)
    
    # Compute deviation from ideal complement sum for each pair (i,j)
    # ideal_partner_demand[i] = ideal_sum[i] - demands[i]
    ideal_partner_demand = ideal_sum[:, None] - demands[None, :]
    
    # Deviation of demands[j] from ideal partner demand for item i
    deviation = np.abs(demands[None, :] - ideal_partner_demand)
    
    # Normalize deviation
    dev_norm = deviation / capacity
    
    # Complementarity score: high when fill ratio is high AND deviation is low
    # Use a combination:
    # - Fill ratio component: strongly prefer near-full bins
    # - Deviation component: prefer pairs close to ideal complement
    
    # Exponential decay on deviation with moderate strength
    dev_score = np.exp(-3.0 * dev_norm)
    
    # Fill ratio with power to emphasize high fills
    fill_score = fill_ratio ** 3
    
    # Combine: product of fill score and deviation score
    heuristic = np.where(valid, fill_score * dev_score, 1e-9)
    
    # Ensure all values are finite and positive
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
