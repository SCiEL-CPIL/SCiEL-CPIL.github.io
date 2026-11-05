import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0

    n = 30
    mid = 15

    # 1. Compute L = lcm(1, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)

    # 2. Compute weights w_i = L // i
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = L // i

    # Left indices 1..15, Right indices 16..30
    left_indices = list(range(1, mid + 1))
    right_indices = list(range(mid + 1, n + 1))
    
    w_left = np.array([w[i] for i in left_indices], dtype=np.int64)
    w_right = np.array([w[i] for i in right_indices], dtype=np.int64)

    # 3. Generate all sums for Left half
    # 3^15 = 14348907
    num_left = 3 ** mid
    # Use iterative mask generation
    # To save memory, we don't store all coeffs, just masks.
    # We need to compute sums.
    
    # Efficient sum computation using NumPy broadcasting
    # Create digit matrix
    # d_{k,j} = (mask // 3^j) % 3
    # This creates a (num_left, 15) array.
    # 14M * 15 * 1 byte = 215 MB. Acceptable.
    
    def gen_sums(weights, num_items, num_masks):
        if time.time() - start_time > deadline:
            return None, None
            
        # Generate masks
        masks = np.arange(num_masks, dtype=np.int64)
        
        # Generate digits
        # powers = np.array([3**j for j in range(num_items)], dtype=np.int64)
        # digits = (masks[:, None] // powers[None, :]) % 3
        # This creates (14M, 15) array.
        
        # To be safe with memory, we can do this in chunks or use a smarter method.
        # But 215MB is usually fine.
        
        powers = np.array([3**j for j in range(num_items)], dtype=np.int64)
        digits = (masks[:, None] // powers[None, :]) % 3
        
        # Map digits 0,1,2 to -1,0,1
        coeffs = digits.astype(np.int8) - 1
        
        # Compute sums
        sums = coeffs @ weights
        
        return sums, masks

    sums_left, masks_left = gen_sums(w_left, mid, num_left)
    
    if time.time() - start_time > deadline:
        # Fallback
        fallback = [0] * n
        fallback[0] = 1
        return fallback

    # Sort left sums
    idx_left = np.argsort(sums_left)
    sorted_sums_left = sums_left[idx_left]
    sorted_masks_left = masks_left[idx_left]
    
    # 4. Generate all sums for Right half
    num_right = 3 ** (n - mid)
    sums_right, masks_right = gen_sums(w_right, n - mid, num_right)
    
    if time.time() - start_time > deadline:
        # Fallback
        fallback = [0] * n
        fallback[0] = 1
        return fallback

    # 5. Find best pair
    # For each right sum, find left sum closest to -right_sum
    targets = -sums_right
    pos = np.searchsorted(sorted_sums_left, targets)
    
    # Candidates: pos and pos-1
    # Handle boundaries
    pos_clamped = np.clip(pos, 1, len(sorted_sums_left) - 1)
    
    cand1_idx = pos_clamped
    cand2_idx = pos_clamped - 1
    
    # Get candidate sums and masks
    cand1_sums = sorted_sums_left[cand1_idx]
    cand2_sums = sorted_sums_left[cand2_idx]
    
    cand1_masks = sorted_masks_left[cand1_idx]
    cand2_masks = sorted_masks_left[cand2_idx]
    
    # Calculate total sums
    total1 = sums_right + cand1_sums
    total2 = sums_right + cand2_sums
    
    # Absolute values
    abs1 = np.abs(total1)
    abs2 = np.abs(total2)
    
    # We need the minimum non-zero absolute value
    # Create a mask for non-zero totals
    valid1 = total1 != 0
    valid2 = total2 != 0
    
    # Set invalid ones to infinity
    abs1[~valid1] = np.inf
    abs2[~valid2] = np.inf
    
    # Find the global minimum
    # We can combine into one array or check separately
    min_abs1 = np.min(abs1)
    min_abs2 = np.min(abs2)
    
    if min_abs1 < min_abs2:
        best_idx_right = np.argmin(abs1)
        best_left_mask = cand1_masks[best_idx_right]
        best_right_mask = masks_right[best_idx_right]
        best_val = min_abs1
    else:
        best_idx_right = np.argmin(abs2)
        best_left_mask = cand2_masks[best_idx_right]
        best_right_mask = masks_right[best_idx_right]
        best_val = min_abs2
        
    if best_val == np.inf:
        # Should not happen, but fallback
        fallback = [0] * n
        fallback[0] = 1
        return fallback

    # 6. Reconstruct coefficients
    result = [0] * n
    
    # Decode left mask
    mask = best_left_mask
    for j, idx in enumerate(left_indices):
        digit = (mask // (3 ** j)) % 3
        result[idx - 1] = digit - 1
        
    # Decode right mask
    mask = best_right_mask
    for j, idx in enumerate(right_indices):
        digit = (mask // (3 ** j)) % 3
        result[idx - 1] = digit - 1
        
    return result
