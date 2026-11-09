"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)
RECS = [1.0 / d for d in DENOMS]
REC_FRACS = [Fraction(1, d) for d in DENOMS]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Split indices into two halves
    mid = N // 2
    left_denoms = DENOMS[:mid]
    right_denoms = DENOMS[mid:]
    left_recs = RECS[:mid]
    right_recs = RECS[mid:]
    
    # Generate subset sums and masks for left half
    # sums_L: list of floats
    # masks_L: list of ints (bitmask of selected items in left half)
    sums_L = [0.0]
    masks_L = [0]
    
    for i, r in enumerate(left_recs):
        # Double the list
        # We iterate backwards to avoid modifying the list while iterating, 
        # or just create new list. Creating new list is safer and often fast enough.
        # Or: sums_L += [s + r for s in sums_L]
        # This is O(N) memory and time per step.
        new_sums = [s + r for s in sums_L]
        new_masks = [m | (1 << i) for m in masks_L]
        sums_L += new_sums
        masks_L += new_masks
        
    # Generate subset sums and masks for right half
    sums_R = [0.0]
    masks_R = [0]
    
    for i, r in enumerate(right_recs):
        new_sums = [s + r for s in sums_R]
        new_masks = [m | (1 << i) for m in masks_R]
        sums_R += new_sums
        masks_R += new_masks
        
    # Sort right half sums and masks together
    # We need to sort by sum, but keep mask associated.
    # Create list of tuples (sum, mask)
    pairs_R = list(zip(sums_R, masks_R))
    pairs_R.sort(key=lambda x: x[0])
    
    sorted_sums_R = [p[0] for p in pairs_R]
    sorted_masks_R = [p[1] for p in pairs_R]
    
    # Find best candidates
    best_deficit = 1.0 # 1 - 0
    best_indices = []
    
    # We will collect top candidates from float search to verify with Fractions
    # Because float precision might be an issue near 1.0
    candidate_indices = []
    
    for s_l, m_l in zip(sums_L, masks_L):
        if time.time() > deadline:
            break
            
        target = 1.0 - s_l
        
        # Find largest sum_R < target
        # bisect_left returns index of first element >= target
        # We want last element < target, so idx - 1
        idx = bisect.bisect_left(sorted_sums_R, target) - 1
        
        if idx >= 0:
            s_r = sorted_sums_R[idx]
            m_r = sorted_masks_R[idx]
            
            total_sum_float = s_l + s_r
            
            # Reconstruct indices
            # Left mask m_l corresponds to left_denoms
            # Right mask m_r corresponds to right_denoms
            inds = []
            for i in range(mid):
                if m_l & (1 << i):
                    inds.append(i)
            for i in range(len(right_denoms)):
                if m_r & (1 << i):
                    inds.append(mid + i)
            
            # Only consider non-empty? 
            # The problem says "Return nonempty".
            # If inds is empty, sum is 0.
            # But we are maximizing sum. 0 is the minimum.
            # We will likely find non-empty ones.
            # However, if we only find empty, we must return something valid.
            # 1/2 is valid.
            
            # Add to candidates
            # We don't know if it's valid yet (exact < 1)
            # Store (total_sum_float, inds)
            candidate_indices.append((total_sum_float, inds))
            
            # Optimization: We can limit the number of candidates
            # The best float sum is the most likely to be the best exact sum.
            # We can keep top 50.
            if len(candidate_indices) > 100:
                # Keep only top 100 by float sum
                candidate_indices.sort(key=lambda x: -x[0])
                candidate_indices = candidate_indices[:100]

    # Verify candidates with Fractions
    # Sort candidates by float sum descending
    candidate_indices.sort(key=lambda x: -x[0])
    
    valid_best_sum = Fraction(0, 1)
    valid_best_indices = []
    
    for s_float, inds in candidate_indices:
        if time.time() > deadline:
            break
            
        # Calculate exact sum
        exact_sum = Fraction(0, 1)
        for i in inds:
            exact_sum += REC_FRACS[i]
            
        if exact_sum < 1:
            # It is valid. Since we sorted by float sum descending,
            # the first valid one is likely the best.
            # However, float ordering might be slightly off.
            # But if float sum is high, exact sum is likely high.
            # To be safe, we can check a few more or just take the first valid one.
            # Given the density, the first valid one from top float candidates should be optimal or very close.
            # Let's take the first valid one found in this sorted list.
            valid_best_sum = exact_sum
            valid_best_indices = inds
            break
            
    # Fallback if no valid candidate found (should not happen)
    if not valid_best_indices:
        # Return index 0 (denominator 2)
        valid_best_indices = [0]
        valid_best_sum = Fraction(1, 2)
        
    return valid_best_indices
# EVOLVE-BLOCK-END
