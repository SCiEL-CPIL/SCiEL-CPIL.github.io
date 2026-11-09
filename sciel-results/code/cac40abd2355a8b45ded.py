import time
from fractions import Fraction
from itertools import combinations
import math

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin
    
    # Denominators are 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    mid = n // 2
    left_denoms = denominators[:mid]
    right_denoms = denominators[mid:]
    
    left_size = len(left_denoms)
    right_size = len(right_denoms)
    
    # Generate all subset sums for the left half
    # Using float for speed
    left_sums = []
    # We need to map mask to indices later, so we store the mask
    # mask is an integer where bit i is set if left_denoms[i] is included
    
    # Optimization: Precompute reciprocals as floats
    left_recs = [1.0 / d for d in left_denoms]
    right_recs = [1.0 / d for d in right_denoms]
    
    # Generate left sums
    for mask in range(1 << left_size):
        s = 0.0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += left_recs[i]
            m >>= 1
            i += 1
        # If sum is already >= 1, it won't be part of a valid solution 
        # combined with any non-negative right sum. 
        # But we can still store it, or prune. Pruning saves memory/time in sort?
        # Actually, if s >= 1, we don't need it because right_sum >= 0 implies total >= 1.
        if s < 1.0:
            left_sums.append((s, mask))
            
    # Sort left sums by sum
    left_sums.sort(key=lambda x: x[0])
    
    # We will binary search on the sums
    left_sum_values = [x[0] for x in left_sums]
    
    # Generate right sums and find best match
    best_deficit = 1.0  # Start with empty set deficit 1.0
    best_left_mask = 0
    best_right_mask = 0
    
    # We will keep a list of top candidates to verify with Fractions
    # Tuple: (float_deficit, left_mask, right_mask)
    top_candidates = []
    
    for mask in range(1 << right_size):
        s_right = 0.0
        m = mask
        i = 0
        while m:
            if m & 1:
                s_right += right_recs[i]
            m >>= 1
            i += 1
            
        if s_right >= 1.0:
            continue
            
        # We need s_left such that s_left + s_right < 1.0
        # i.e., s_left < 1.0 - s_right
        target = 1.0 - s_right
        
        # Binary search for the largest s_left < target
        # left_sum_values is sorted ascending
        # We want the rightmost element < target
        lo, hi = 0, len(left_sum_values)
        # Find insertion point
        # bisect_left returns index where target would be inserted to maintain order
        # We want the element before that, but strictly less than target.
        # If target is 0.5, and list has 0.4, 0.6, bisect_left(0.5) is 1. Element at 0 is 0.4.
        # If target is 0.5, and list has 0.5, 0.6, bisect_left(0.5) is 0. No element < 0.5? 
        # Wait, bisect_left finds position to insert. 
        # If list is [0.4, 0.5, 0.6] and target is 0.5.
        # bisect_left(0.5) -> 1. Index 0 is 0.4 which is < 0.5.
        # If list is [0.4, 0.6] and target is 0.5.
        # bisect_left(0.5) -> 1. Index 0 is 0.4 which is < 0.5.
        # If list is [0.5, 0.6] and target is 0.5.
        # bisect_left(0.5) -> 0. No index before 0. So no valid s_left?
        # But s_left must be < 0.5. 0.5 is not < 0.5. So correct.
        
        idx = 0
        # Use bisect module for speed
        import bisect
        pos = bisect.bisect_left(left_sum_values, target)
        if pos > 0:
            idx = pos - 1
            s_left = left_sum_values[idx]
            l_mask = left_sums[idx][1]
            
            total_sum = s_left + s_right
            deficit = 1.0 - total_sum
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = l_mask
                best_right_mask = mask
                
            # Add to top candidates if it's a good one
            # We only care about very small deficits
            if deficit < 1e-6: # Threshold for exact verification
                top_candidates.append((deficit, l_mask, mask))
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    # If we broke early, we might have missed some, but we have a best so far.
    # Now, verify the top candidates with Fraction to ensure exactness.
    
    # Also verify the current best
    candidates_to_check = set()
    candidates_to_check.add((best_left_mask, best_right_mask))
    
    # Add top candidates
    # Sort top_candidates by deficit to check best ones first
    top_candidates.sort()
    for d, lm, rm in top_candidates[:100]:
        candidates_to_check.add((lm, rm))
        
    # Also add empty set
    candidates_to_check.add((0, 0))
    
    best_exact_deficit = None
    best_exact_l_mask = 0
    best_exact_r_mask = 0
    
    # Precompute Fraction reciprocals
    left_frac_recs = [Fraction(1, d) for d in left_denoms]
    right_frac_recs = [Fraction(1, d) for d in right_denoms]
    
    for lm, rm in candidates_to_check:
        s_left = Fraction(0)
        m = lm
        i = 0
        while m:
            if m & 1:
                s_left += left_frac_recs[i]
            m >>= 1
            i += 1
            
        s_right = Fraction(0)
        m = rm
        i = 0
        while m:
            if m & 1:
                s_right += right_frac_recs[i]
            m >>= 1
            i += 1
            
        total = s_left + s_right
        if total < 1:
            deficit = 1 - total
            if best_exact_deficit is None or deficit < best_exact_deficit:
                best_exact_deficit = deficit
                best_exact_l_mask = lm
                best_exact_r_mask = rm
                
    # Construct indices
    indices = []
    m = best_exact_l_mask
    i = 0
    while m:
        if m & 1:
            indices.append(i) # Index in left_denoms, which is same as global index
        m >>= 1
        i += 1
        
    m = best_exact_r_mask
    i = 0
    while m:
        if m & 1:
            indices.append(mid + i) # Index in right_denoms, offset by mid
        m >>= 1
        i += 1
        
    if not indices:
        # Should not happen as empty set is valid and deficit 1.0
        # But if somehow all sums >= 1, we return empty? 
        # Task says "Return nonempty distinct zero-based indices".
        # "Reciprocal sum must be STRICTLY BELOW 1".
        # If empty set is the only one? Sum 0 < 1. Deficit 1.
        # But task says "Return nonempty".
        # However, the seed program returns a non-empty list.
        # If the optimal is empty, we must return a non-empty set with sum < 1.
        # The smallest denominator is 2, sum 0.5 < 1. Deficit 0.5.
        # So we can always pick index 0.
        indices = [0]
        
    return indices
