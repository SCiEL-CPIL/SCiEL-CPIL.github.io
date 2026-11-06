"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    mid = n // 2
    left_denoms = denominators[:mid]
    right_denoms = denominators[mid:]
    
    left_size = len(left_denoms)
    right_size = len(right_denoms)
    
    # Generate all subset sums for the right half
    # Store as (float_sum, bitmask, indices_list)
    # To save memory and time, we can store just sum and mask, and reconstruct indices later if needed
    # But we need to return indices. Reconstructing from mask is fast.
    
    right_sums = []
    # Iterate through all masks for the right half
    for mask in range(1 << right_size):
        s = 0.0
        # Calculate sum
        # Optimization: Use DP or simple loop. Simple loop is fine for 24 bits.
        # Actually, generating all sums can be done more efficiently.
        # Let's just do the loop for clarity and correctness first.
        # If too slow, we can optimize generation.
        # 2^24 = 16,777,216. Looping 24 times is ~400M ops. Might be tight in Python.
        # Let's try to generate sums incrementally.
        
    # Better generation strategy:
    # Start with [0.0]
    # For each denom, append existing sums + 1/denom
    # This is O(2^k * k) as well but has better cache locality?
    # Let's stick to the mask loop but optimize the inner sum calculation.
    # Or use the incremental list approach.
    
    right_sums = [(0.0, 0)]
    for i, d in enumerate(right_denoms):
        inv_d = 1.0 / d
        new_sums = []
        for s, m in right_sums:
            new_sums.append((s + inv_d, m | (1 << i)))
        right_sums.extend(new_sums)
        
    # Sort right_sums by sum
    # We need to be careful with float precision.
    # Sort by float sum.
    right_sums.sort(key=lambda x: x[0])
    
    # Extract sorted sums for binary search
    # We keep the list of (sum, mask)
    
    # Now iterate through left half subsets
    best_deficit = float('inf')
    best_indices = []
    
    # We will generate left sums on the fly or pre-compute?
    # Pre-computing left sums is similar cost.
    # Let's generate left sums and check against right_sums.
    
    left_sums = [(0.0, 0)]
    for i, d in enumerate(left_denoms):
        inv_d = 1.0 / d
        new_sums = []
        for s, m in left_sums:
            new_sums.append((s + inv_d, m | (1 << i)))
        left_sums.extend(new_sums)
        
    # To avoid iterating over all left sums if we find a very good one, 
    # we could sort left sums too and do a two-pointer or similar, 
    # but binary search is O(N log M) which is fine.
    
    # However, float precision might cause issues near 1.0.
    # We will collect the top K candidates from the float search and verify with Fraction.
    
    candidates = []
    
    for s_l, mask_l in left_sums:
        # Target: s_r < 1.0 - s_l
        target = 1.0 - s_l
        
        # Binary search in right_sums for largest s_r < target
        # right_sums is sorted by s_r
        # We want the rightmost index where sum < target
        # bisect_left returns the first index where sum >= target
        # So we want index - 1
        idx = bisect.bisect_left([x[0] for x in right_sums], target)
        
        # This list comprehension [x[0] for x in right_sums] is expensive inside the loop.
        # Pre-extract right sums into a separate list for bisect.
        
    # Refactor: Pre-extract right sums
    right_sum_vals = [x[0] for x in right_sums]
    
    for s_l, mask_l in left_sums:
        target = 1.0 - s_l
        
        # Find insertion point
        idx = bisect.bisect_left(right_sum_vals, target)
        
        # The best candidate is at idx - 1
        if idx > 0:
            s_r, mask_r = right_sums[idx - 1]
            total_float = s_l + s_r
            if total_float < 1.0:
                # Potential candidate
                # Add to candidates list for verification
                # We only need to keep the best few
                # Deficit approx: 1.0 - total_float
                deficit_approx = 1.0 - total_float
                candidates.append((deficit_approx, mask_l, mask_r))
                
        # Also check idx itself? No, bisect_left finds first >= target.
        # So idx is the first one that is too big (or equal).
        # We want strictly below 1.
        # If s_r == target, then s_l + s_r == 1.0, which is not allowed.
        # So we must pick the one before.
        # If idx == 0, no valid s_r in right half for this s_l.
        
    # Sort candidates by deficit_approx
    candidates.sort(key=lambda x: x[0])
    
    # Verify top candidates with Fraction
    # Limit verification to top 20 to save time
    for deficit_approx, mask_l, mask_r in candidates[:20]:
        # Reconstruct indices
        indices = []
        # Left indices
        for i in range(left_size):
            if mask_l & (1 << i):
                indices.append(i)
        # Right indices
        for i in range(right_size):
            if mask_r & (1 << i):
                indices.append(mid + i)
        
        if not indices:
            continue
            
        # Calculate exact sum
        total_frac = Fraction(0)
        for idx in indices:
            total_frac += Fraction(1, denominators[idx])
            
        if total_frac < 1:
            exact_deficit = 1 - total_frac
            # Update best
            if best_deficit == float('inf') or exact_deficit < best_deficit:
                best_deficit = exact_deficit
                best_indices = indices
                
        if time.time() > deadline:
            break
            
    if not best_indices:
        # Fallback: return empty or a known valid small set if no candidate found
        # This shouldn't happen, but just in case
        # Greedy fallback
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
        
    return best_indices
# EVOLVE-BLOCK-END
