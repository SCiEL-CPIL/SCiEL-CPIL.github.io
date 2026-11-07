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
    
    # Count how many items k (k != i, k != j) fit into the residual space
    # residual[i, j] = gap[i, j]
    # count_fitting[i, j] = sum over k of (1 if d[k] <= gap[i, j] and k != i and k != j)
    
    # For each pair (i, j), we want to count items k such that d[k] <= gap[i, j]
    # and k is not i and not j.
    
    # First, compute for each possible gap value g, how many items have size <= g.
    # Since d is a fixed array, we can precompute a cumulative count.
    # But gap[i, j] varies continuously (well, discretely based on d values).
    # We can vectorize this: for each (i, j), count = sum(d <= gap[i, j]) - (1 if d[i] <= gap[i, j]) - (1 if d[j] <= gap[i, j])
    
    # Let's compute base_count[i, j] = sum_k (d[k] <= gap[i, j])
    # This is equivalent to: for each row of gap, count how many d values are <= that gap.
    # We can use searchsorted on sorted d.
    
    d_sorted = np.sort(d)
    # For each gap[i, j], find the number of elements in d_sorted that are <= gap[i, j]
    # searchsorted with side='right' gives the index where gap[i,j] would be inserted to keep order,
    # which equals the count of elements <= gap[i, j] if all are distinct, but with duplicates it's the count of elements < gap[i,j] + count of elements == gap[i,j]?
    # Actually, np.searchsorted(d_sorted, gap, side='right') returns the number of elements <= gap.
    
    # gap is an (n, n) array. d_sorted is (n,).
    # We need to broadcast d_sorted against gap.
    # This might be memory intensive for n=120 (120*120*8 bytes = 115KB for gap, similar for searchsorted result).
    # Let's try.
    
    base_counts = np.searchsorted(d_sorted, gap, side='right')
    
    # Now subtract 1 if d[i] <= gap[i, j] and 1 if d[j] <= gap[i, j]
    # Note: d[i] <= gap[i, j] is equivalent to d[i] <= capacity - d[i] - d[j] => 2*d[i] + d[j] <= capacity
    # But simpler: just check the condition directly.
    
    # cond_i[i, j] = (d[i] <= gap[i, j])
    # cond_j[i, j] = (d[j] <= gap[i, j])
    
    # d[i] is d[:, None], d[j] is d[None, :]
    cond_i = d[:, None] <= gap
    cond_j = d[None, :] <= gap
    
    # count_fitting[i, j] = base_counts[i, j] - cond_i[i, j] - cond_j[i, j]
    # But we must ensure count_fitting is not negative.
    count_fitting = base_counts - cond_i.astype(np.int64) - cond_j.astype(np.int64)
    count_fitting = np.maximum(count_fitting, 0)
    
    # The heuristic should be inversely proportional to count_fitting.
    # If count_fitting is 0, it means no other item fits in the gap, which is ideal (dead gap).
    # We want a high heuristic for low count_fitting.
    # Let's use 1 / (count_fitting + 1).
    # This gives 1.0 for count=0, 0.5 for count=1, etc.
    
    # We also want to combine this with the original gap-based heuristic?
    # The directive says "inversely proportional to the count".
    # Let's try a blend or just the new one.
    # The original was 1/(gap+1).
    # Let's try H = 1 / (count_fitting + 1) * (1 / (gap + 1)) ?
    # Or just H = 1 / (count_fitting + 1).
    # Let's stick to the directive: "inversely proportional to the count".
    
    # However, if count is the same for many pairs, we might want to break ties with gap.
    # Let's use H = 1 / (count_fitting + 1 + gap * 0.01) ?
    # Or simpler: H = 1 / (count_fitting + 1).
    
    # Let's initialize H with the new heuristic.
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    # For valid pairs, compute the new heuristic.
    safe_count = count_fitting[valid_mask].astype(np.float64)
    candidate_H = 1.0 / (safe_count + 1.0)
    
    # Apply the mask
    H[valid_mask] = candidate_H
    
    # Ensure no non-finite values exist (defensive check)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Check deadline guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        pass
        
    return H
# EVOLVE-BLOCK-END
