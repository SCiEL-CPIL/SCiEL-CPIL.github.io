"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    - We favor pairs (i, j) that have a combined size close to the bin capacity.
    - Such pairs 'saturate' the bin, reducing the remaining space for other items
      and effectively closing the bin sooner, which helps minimize the total
      number of bins used.
    - The heuristic is defined as 1 / (remaining_space + 1) for valid pairs,
      where remaining_space = capacity - (demand[i] + demand[j]).
    - Invalid pairs (where demand[i] + demand[j] > capacity) or diagonal entries
      are set to a very small value (1e-9) to make them effectively unselectable.
    """
    start_time = time.time()
    
    # Convert to numpy array for vectorized operations
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    # Compute the sum of demands for all pairs (i, j)
    # pair_sums[i, j] = d[i] + d[j]
    pair_sums = d[:, None] + d[None, :]
    
    # Compute the gap (remaining capacity) for all pairs
    # gap[i, j] = capacity - pair_sums[i, j]
    gap = capacity - pair_sums
    
    # Identify valid pairs:
    # 1. The pair must fit in the bin: gap >= 0
    # 2. It must not be the same item (i != j)
    valid_mask = (gap >= 0)
    # Ensure diagonal is excluded from valid calculation to be strict
    np.fill_diagonal(valid_mask, False)
    
    # Create a safe denominator: where valid, use gap+1; otherwise, use 1 (to avoid 0 div)
    safe_denom = np.where(valid_mask, gap + 1.0, 1.0)
    
    # Calculate candidate heuristic values
    # 1.0 / safe_denom will be large for small gaps (valid tight fits)
    # and 1.0 for invalid pairs (which we will mask out).
    candidate_H = 1.0 / safe_denom
    
    # Apply the mask: keep calculated values for valid pairs, 1e-9 for others
    H = np.where(valid_mask, candidate_H, 1e-9)
    
    # Ensure no non-finite values exist (defensive check)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Check deadline guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H
        
    return H
# EVOLVE-BLOCK-END
