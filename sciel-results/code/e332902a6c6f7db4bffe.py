"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # 1. Base Heuristic: Reward tight fits
    # pair_sum[i, j] = d[i] + d[j]
    pair_sum = d[:, None] + d[None, :]
    waste = capacity - pair_sum
    
    # Base score: inverse waste, sharpened by power
    # High power ensures that only very tight fits get high scores
    power = 8.0
    # Use where to avoid division by zero or negative waste issues
    # If waste >= 0, it fits.
    base = np.where(waste >= 0, 1.0 / (waste + 1.0), 0.0)
    base = np.power(base, power)
    
    # 2. Complementarity Heuristic: Identify ideal partners
    # Sort items to find best complements efficiently
    sorted_idx = np.argsort(d)
    sorted_d = d[sorted_idx]
    
    # ideal_partner_sorted[i] is the index in sorted_d of the best partner for sorted_d[i]
    # Best partner is the largest item that fits with sorted_d[i]
    ideal_partner_sorted = np.zeros(n, dtype=np.int64)
    
    # Two-pointer technique
    # j starts at the end (largest items)
    j = n - 1
    for i in range(n):
        # Move j left until sorted_d[i] + sorted_d[j] <= capacity
        while j > i and sorted_d[i] + sorted_d[j] > capacity:
            j -= 1
        # If j <= i, no partner found in the "larger" part (or self)
        # Actually, if j < i, it means no item from i+1..n-1 fits.
        # But we can also pair with items < i? 
        # The two-pointer logic above finds the largest j > i that fits.
        # What if the best partner is smaller than i?
        # Since we are looking for "tight fit", and d is sorted ascending,
        # the tightest fit for d[i] is usually the largest possible d[j] such that sum <= cap.
        # If j ends up < i, it means even the item at i (itself, if we allowed it) 
        # or larger ones don't fit. But d[i] + d[i] might fit?
        # The problem allows placing item j after item i. i != j usually in bin packing context?
        # "placing item j right after item i in the same bin". 
        # Usually distinct items. 
        # If j < i, it means no item in (i, n) fits.
        # We should check if any item in (0, i) fits?
        # If d[i] is small, its best partner is large (high index).
        # If d[i] is large, its best partner is small (low index).
        # The two-pointer sweep from left (small i) to right (large i) with j moving left
        # correctly identifies the largest j for each i.
        # If j < i, it means no j > i fits.
        # Does a j < i fit? 
        # If d[i] + d[j] <= capacity for some j < i, then d[i] + d[i] <= 2*d[i] <= capacity?
        # Not necessarily.
        # However, for the "ideal partner" heuristic, we want the item that completes the bin best.
        # If no j > i fits, the best partner might be a smaller item.
        # But given the sorted order, if the largest item that fits with d[i] is smaller than d[i],
        # then d[i] is quite large.
        # Let's just record the best j found. If j < i, we might still have a valid partner if j >= 0.
        # But wait, if j < i, the loop `while j > i` stops.
        # We need to ensure we find the *best* partner.
        # For small i, j is large. For large i, j is small.
        # The standard two-pointer for "closest sum to capacity" works.
        
        if j >= 0:
            ideal_partner_sorted[i] = j
        else:
            ideal_partner_sorted[i] = i # Fallback to self, will be masked later if i!=j required or by fit check
            
    # Map back to original indices
    # For each original item i, we need to know its ideal partner in original indices.
    # pos_in_sorted[orig_idx] = sorted_pos
    pos_in_sorted = np.zeros(n, dtype=np.int64)
    for k in range(n):
        pos_in_sorted[sorted_idx[k]] = k
        
    ideal_partner_orig = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pos_i = pos_in_sorted[i]
        partner_pos = ideal_partner_sorted[pos_i]
        partner_orig = sorted_idx[partner_pos]
        ideal_partner_orig[i] = partner_orig
        
    # 3. Combine Base and Complementarity
    result = base.copy()
    
    # Add a large bonus for the ideal partner pairs
    # This strongly biases the ACO towards these pairs.
    bonus = 500.0
    
    for i in range(n):
        j = ideal_partner_orig[i]
        # Only add bonus if the pair actually fits and i != j (if distinct items required)
        # The task says "placing item j right after item i". 
        # In bin packing, you don't place an item after itself.
        # But the matrix is n x n. Diagonal entries are usually irrelevant or set to 0/low.
        # If d[i] * 2 <= capacity, i and i could theoretically be "paired" in the heuristic sense,
        # but in the ACO loop, `current` is the last placed item. The next item `j` must be different.
        # The ACO code likely handles `j != current` or `j` is from the remaining set.
        # We should ensure we don't boost the diagonal if it's not useful, or if it's invalid.
        # However, boosting the diagonal doesn't hurt if the ACO excludes it.
        # But if the ACO allows it (e.g. duplicate items?), it might be good.
        # Assuming standard bin packing: distinct items.
        # Let's check fit explicitly.
        if d[i] + d[j] <= capacity and i != j:
            result[i, j] += bonus
            
    # 4. Cleanup
    # Ensure non-finite values are replaced
    result = np.where(np.isfinite(result), result, 1e-9)
    
    # Ensure all non-positive become 1e-9
    # Note: base is 0 for non-fitting. 0 + 0 = 0. We want 1e-9.
    result = np.where(result > 0, result, 1e-9)
    
    # Final check for time
    if time.time() > deadline:
         # Return whatever we have, it should be valid
         pass
         
    return result
# EVOLVE-BLOCK-END
