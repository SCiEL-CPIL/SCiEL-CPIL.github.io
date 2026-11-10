import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85  # Leave 5s margin
    
    D = 2329089562800  # LCM(1..30)
    half = 15
    
    # Weights for first half (indices 0..14, corresponding to 1/1 .. 1/15)
    w1 = np.array([D // (i + 1) for i in range(half)], dtype=np.int64)
    # Weights for second half (indices 15..29, corresponding to 1/16 .. 1/30)
    w2 = np.array([D // (i + 1) for i in range(half, 30)], dtype=np.int64)
    
    # Enumerate sums for first half
    sums1 = np.array([0], dtype=np.int64)
    for w in w1:
        # Create new array of size 3 * len(sums1)
        # sums1 - w, sums1, sums1 + w
        new_sums = np.empty(len(sums1) * 3, dtype=np.int64)
        new_sums[::3] = sums1 - w
        new_sums[1::3] = sums1
        new_sums[2::3] = sums1 + w
        sums1 = new_sums
        
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]
        
    # Enumerate sums for second half
    sums2 = np.array([0], dtype=np.int64)
    for w in w2:
        new_sums = np.empty(len(sums2) * 3, dtype=np.int64)
        new_sums[::3] = sums2 - w
        new_sums[1::3] = sums2
        new_sums[2::3] = sums2 + w
        sums2 = new_sums
        
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]
        
    # Sort sums2 for binary search
    sums2.sort()
    
    # For each sum in sums1, find the closest sum in sums2 such that sum1 + sum2 is close to 0
    # i.e., sum2 is close to -sum1
    targets = -sums1
    
    # searchsorted returns the index where the target would be inserted
    # to maintain sorted order
    pos = np.searchsorted(sums2, targets)
    
    # The closest value is either at pos or pos-1
    # Handle boundaries
    pos1 = np.clip(pos, 0, len(sums2) - 1)
    pos2 = np.clip(pos - 1, 0, len(sums2) - 1)
    
    cand1 = sums1 + sums2[pos1]
    cand2 = sums1 + sums2[pos2]
    
    abs_cand1 = np.abs(cand1)
    abs_cand2 = np.abs(cand2)
    
    # We want to minimize |cand|, but cand != 0
    # Set 0 to a large value so it's not picked
    max_int = np.iinfo(np.int64).max
    abs_cand1[cand1 == 0] = max_int
    abs_cand2[cand2 == 0] = max_int
    
    best_abs = np.minimum(abs_cand1, abs_cand2)
    best_idx = int(np.argmin(best_abs))
    
    if time.time() - start_time > DEADLINE:
        return [0] * 28 + [1, -1]
        
    # Reconstruct coefficients
    # Determine which candidate was chosen
    if abs_cand1[best_idx] <= abs_cand2[best_idx]:
        orig_i = best_idx
        orig_j = int(pos1[best_idx])
    else:
        orig_i = best_idx
        orig_j = int(pos2[best_idx])
        
    coeffs = [0] * 30
    
    # Decode orig_i to get coeffs for first half
    temp = orig_i
    for k in range(half):
        d = temp % 3
        temp //= 3
        coeffs[k] = d - 1
        
    # Decode orig_j to get coeffs for second half
    temp = orig_j
    for k in range(half):
        d = temp % 3
        temp //= 3
        coeffs[half + k] = d - 1
        
    # Verify non-zero
    total_num = sum(coeffs[i] * (D // (i + 1)) for i in range(30))
    if total_num == 0:
        return [0] * 28 + [1, -1]
        
    return coeffs
