import time
from fractions import Fraction
import bisect
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    n1 = n // 2  # 23
    n2 = n - n1  # 24

    def gen_subset_sums_numpy(dens):
        sums = np.array([0.0])
        for i in range(len(dens)):
            val = 1.0 / dens[i]
            sums = np.concatenate([sums, sums + val])
        return sums

    sums1 = gen_subset_sums_numpy(denominators[:n1])
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    sums2 = gen_subset_sums_numpy(denominators[n1:])
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    sums2_sorted = np.sort(sums2)
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    # For each s1, find best s2
    # Vectorize where possible
    # s1 values where s1 < 1
    valid1 = sums1 < 1.0
    s1_valid = sums1[valid1]
    targets = 1.0 - s1_valid
    
    # For each target, find rightmost s2 < target
    # np.searchsorted with side='left' gives insertion point
    # We want the index before that
    pos = np.searchsorted(sums2_sorted, targets, side='left')
    # pos[i] is the index in sums2_sorted where target would be inserted
    # We want sums2_sorted[pos[i] - 1] if pos[i] > 0
    
    # Handle pos=0 case: no valid s2
    valid_pos = pos > 0
    if not np.any(valid_pos):
        return greedy_fallback(denominators)
    
    pos_valid = pos[valid_pos]
    s2_candidates = sums2_sorted[pos_valid - 1]
    s1_candidates = s1_valid[valid_pos]
    
    totals = s1_candidates + s2_candidates
    valid_total = totals < 1.0
    if not np.any(valid_total):
        return greedy_fallback(denominators)
    
    totals_valid = totals[valid_total]
    s1_final = s1_candidates[valid_total]
    s2_final = s2_candidates[valid_total]
    
    deficits = 1.0 - totals_valid
    best_idx = np.argmin(deficits)
    best_deficit = deficits[best_idx]
    best_s1 = s1_final[best_idx]
    best_s2 = s2_final[best_idx]
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    # Now recover masks for best_s1 and best_s2
    # This is the hard part. We need to find which mask gives best_s1 in sums1
    # and which mask gives best_s2 in sums2.
    
    # Since we used NumPy, we lost the mask information.
    # We need to either track masks or use a different approach.
    
    # Option 1: Store (sum, mask) pairs. But that's memory intensive.
    # Option 2: Reconstruct the mask from the sum. But floating point makes this hard.
    # Option 3: Use a different representation.
    
    # Let me use a hybrid: generate sums with masks for the smaller half, 
    # and for the larger half, use a dictionary from sum to mask (but floats as keys are problematic).
    
    # Actually, let me just do the mask tracking in Python for the final verification.
    # The key insight: we have best_s1 and best_s2 as floats. We need to find the exact subsets.
    
    # Better approach: do the entire search with masks, but use NumPy for the initial 
    # candidate generation, then verify.
    
    # Let me redo: generate sums1 and sums2 as before, but also keep track of masks
    # using a different method.
    
    # For 23 items, I can generate all masks and compute sums in a vectorized way.
    # mask i has bits set at positions where the bit is 1.
    # sum = sum over j of (bit_j of mask) * (1/dens[j])
    
    # This is a matrix-vector product!
    # Create a 2^23 x 23 matrix of bits, multiply by 1/dens vector.
    
    def gen_sums_with_masks(dens):
        m = len(dens)
        size = 1 << m
        # Create bit matrix
        # For memory efficiency, process in chunks
        # For 23 items, 8M x 23 = 184M bits = 23MB if packed, or 8M*23*8 = 1.5GB if float.
        # Too much. Let's use a different approach.
        
        # Iterative with masks
        sums = np.array([0.0])
        masks = np.array([0])
        for i in range(m):
            val = 1.0 / dens[i]
            bit = 1 << i
            sums = np.concatenate([sums, sums + val])
            masks = np.concatenate([masks, masks | bit])
        return sums, masks

    sums1, masks1 = gen_sums_with_masks(denominators[:n1])
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    sums2, masks2 = gen_sums_with_masks(denominators[n1:])
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    # Sort sums2 with corresponding masks
    order = np.argsort(sums2)
    sums2_sorted = sums2[order]
    masks2_sorted = masks2[order]
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    # Find best
    valid1 = sums1 < 1.0
    s1_valid = sums1[valid1]
    masks1_valid = masks1[valid1]
    targets = 1.0 - s1_valid
    
    pos = np.searchsorted(sums2_sorted, targets, side='left')
    valid_pos = pos > 0
    if not np.any(valid_pos):
        return greedy_fallback(denominators)
    
    pos_valid = pos[valid_pos]
    s2_candidates = sums2_sorted[pos_valid - 1]
    m2_candidates = masks2_sorted[pos_valid - 1]
    s1_candidates = s1_valid[valid_pos]
    m1_candidates = masks1_valid[valid_pos]
    
    totals = s1_candidates + s2_candidates
    valid_total = totals < 1.0
    if not np.any(valid_total):
        return greedy_fallback(denominators)
    
    totals_valid = totals[valid_total]
    s1_final = s1_candidates[valid_total]
    s2_final = s2_candidates[valid_total]
    m1_final = m1_candidates[valid_total]
    m2_final = m2_candidates[valid_total]
    
    deficits = 1.0 - totals_valid
    best_idx = np.argmin(deficits)
    best_mask1 = m1_final[best_idx]
    best_mask2 = m2_final[best_idx]
    
    if time.time() - start_time > DEADLINE:
        return greedy_fallback(denominators)

    # Convert masks to indices
    selected = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            selected.append(i)
    for i in range(n2):
        if best_mask2 & (1 << i):
            selected.append(n1 + i)

    # Verify with Fraction
    if not selected:
        return greedy_fallback(denominators)

    total_frac = Fraction()
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])
    
    if total_frac >= 1:
        return greedy_fallback(denominators)

    return selected

def greedy_fallback(denominators):
    selected = []
    total = Fraction()
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
