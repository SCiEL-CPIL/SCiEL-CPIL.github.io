import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin

    denominators = list(range(2, 49))
    n = len(denominators)  # 47
    
    # Split into two halves
    # 23 and 24 items
    n1 = 23
    n2 = n - n1
    
    left_dens = denominators[:n1]
    right_dens = denominators[n1:]

    def generate_sums_with_pruning(dens):
        """
        Generates all subset sums < 1.0 and their corresponding masks.
        Returns numpy arrays for sums and masks.
        """
        sums = [0.0]
        masks = [0]
        
        for i, d in enumerate(dens):
            val = 1.0 / d
            # Create new entries by adding val to existing sums
            # We need to filter out sums >= 1.0
            # Since sums are not sorted, we can't easily prune during generation 
            # without sorting, but sorting at every step is expensive.
            # However, the number of valid sums grows exponentially but is bounded by 1/min_val.
            # For now, let's just generate all and filter at the end of the loop?
            # Or filter immediately.
            
            new_sums = []
            new_masks = []
            bit = 1 << i
            
            for s, m in zip(sums, masks):
                new_s = s + val
                if new_s < 1.0:
                    new_sums.append(new_s)
                    new_masks.append(m | bit)
            
            sums += new_sums
            masks += new_masks
            
            # Optional: Check time
            if time.time() - start_time > DEADLINE:
                # Fallback to a simpler heuristic if we run out of time
                return None, None
        
        return np.array(sums, dtype=np.float64), np.array(masks, dtype=np.int64)

    # Generate for left half
    sums1, masks1 = generate_sums_with_pruning(left_dens)
    if sums1 is None:
        return greedy_fallback(denominators)

    # Generate for right half
    sums2, masks2 = generate_sums_with_pruning(right_dens)
    if sums2 is None:
        return greedy_fallback(denominators)

    # Sort sums2 and masks2
    # We need to sort sums2 and keep masks2 aligned
    order = np.argsort(sums2)
    sums2_sorted = sums2[order]
    masks2_sorted = masks2[order]

    best_deficit = float('inf')
    best_mask1 = 0
    best_mask2 = 0

    # For each sum in sums1, find best sum in sums2_sorted
    # We want s1 + s2 < 1.0
    # s2 < 1.0 - s1
    # Use searchsorted to find the largest s2 < 1.0 - s1
    
    # Vectorize the target calculation
    # targets = 1.0 - sums1
    # But we need to handle the case where s1 >= 1.0 (should be pruned, but just in case)
    # Also, we need to find the index in sums2_sorted.
    
    # np.searchsorted returns the index where the target would be inserted to maintain order.
    # If side='left', it returns the first index where sums2_sorted[i] >= target.
    # We want the largest index j such that sums2_sorted[j] < target.
    # This is pos - 1, where pos = searchsorted(target, side='left').
    
    targets = 1.0 - sums1
    # Only consider s1 < 1.0
    valid1 = sums1 < 1.0
    if not np.any(valid1):
        return greedy_fallback(denominators)
        
    s1_valid = sums1[valid1]
    m1_valid = masks1[valid1]
    targets_valid = targets[valid1]
    
    # searchsorted
    pos = np.searchsorted(sums2_sorted, targets_valid, side='left')
    
    # pos[i] is the index in sums2_sorted.
    # We want pos[i] - 1.
    # If pos[i] == 0, there is no valid s2 (since sums2 are >= 0, and target > 0, 
    # if pos=0 it means sums2_sorted[0] >= target, so no s2 < target? 
    # Wait. If target is very small, e.g., 0.01, and min(sums2) is 0.02, pos=0. 
    # Then s2 must be < 0.01. If 0 is in sums2 (empty set), then s2=0 is valid.
    # 0 is always in sums2.
    # sums2_sorted[0] is 0.0 (empty set).
    # If target > 0, pos will be at least 1?
    # searchsorted([0, 0.1, 0.2], 0.05, side='left') -> 1.
    # pos-1 = 0. sums2_sorted[0] = 0.0. 0.0 < 0.05. Valid.
    # If target <= 0, pos=0. But s1 < 1.0 implies target > 0.
    # So pos should always be >= 1 if 0 is in sums2.
    
    valid_pos = pos > 0
    if not np.any(valid_pos):
        return greedy_fallback(denominators)
        
    pos_valid = pos[valid_pos]
    s2_indices = pos_valid - 1
    
    s2_candidates = sums2_sorted[s2_indices]
    m2_candidates = masks2_sorted[s2_indices]
    
    s1_candidates = s1_valid[valid_pos]
    m1_candidates = m1_valid[valid_pos]
    
    totals = s1_candidates + s2_candidates
    # Double check strict inequality
    valid_total = totals < 1.0
    
    if not np.any(valid_total):
        return greedy_fallback(denominators)
        
    totals_valid = totals[valid_total]
    deficits = 1.0 - totals_valid
    
    best_idx = np.argmin(deficits)
    best_deficit = deficits[best_idx]
    best_mask1 = m1_candidates[best_idx]
    best_mask2 = m2_candidates[best_idx]

    # Reconstruct indices
    selected = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            selected.append(i)
    for i in range(n2):
        if best_mask2 & (1 << i):
            selected.append(n1 + i)

    if not selected:
        return greedy_fallback(denominators)

    # Verify with Fraction
    total_frac = Fraction(0)
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])
        
    if total_frac >= 1:
        return greedy_fallback(denominators)

    return selected

def greedy_fallback(denominators):
    selected = []
    total = Fraction(0)
    # Sort by denominator descending to add smaller fractions first?
    # Or just iterate.
    # Greedy: add if sum < 1.
    # To minimize deficit, we want to get close to 1.
    # Adding smaller fractions (larger denominators) allows more flexibility.
    # Let's try adding in order of increasing value (decreasing denominator) 
    # but stopping when sum >= 1.
    # Actually, a simple greedy on sorted denominators might not be optimal, 
    # but it's a valid fallback.
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
