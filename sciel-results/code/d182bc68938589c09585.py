import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 80.0  # Leave 10s margin for 90s limit
    
    n = 30
    # Calculate LCM of 1..30
    H = 1
    for i in range(1, n + 1):
        H = H * i // math.gcd(H, i)
        
    # Weights W_i = H / i
    W = [H // i for i in range(1, n + 1)]
    
    mid = n // 2  # 15
    
    def generate_half(start, end):
        """Generates numpy arrays of sums and masks for indices start to end-1."""
        length = end - start
        sums = np.array([0], dtype=np.int64)
        masks = np.array([0], dtype=np.int64)
        
        for i in range(length):
            w = W[start + i]
            mult = 3 ** i
            
            # Expand sums: [s - w, s, s + w]
            # Expand masks: [m, m + mult, m + 2*mult]
            # Note: The order must be consistent.
            # c = -1 -> sum - w, mask + 0 (digit 0)
            # c = 0  -> sum    , mask + mult (digit 1)
            # c = 1  -> sum + w, mask + 2*mult (digit 2)
            
            # Using np.concatenate is fast
            new_sums = np.concatenate([sums - w, sums, sums + w])
            new_masks = np.concatenate([masks, masks + mult, masks + 2 * mult])
            
            sums = new_sums
            masks = new_masks
            
        return sums, masks

    # Generate Half A (0 to 14)
    sums_a, masks_a = generate_half(0, mid)
    
    # Generate Half B (15 to 29)
    sums_b, masks_b = generate_half(mid, n)
    
    # Sort Half B by sum
    # We need to sort sums_b and keep masks_b aligned
    sort_idx = np.argsort(sums_b)
    sorted_sums_b = sums_b[sort_idx]
    sorted_masks_b = masks_b[sort_idx]
    
    best_val = np.iinfo(np.int64).max
    best_mask_a = -1
    best_mask_b = -1
    
    # Search for best pair
    # Iterate through A
    for i in range(len(sums_a)):
        if time.time() - start_time > deadline:
            break
            
        a_sum = sums_a[i]
        target = -a_sum
        
        # Find insertion point in sorted_sums_b
        idx = np.searchsorted(sorted_sums_b, target)
        
        # Check idx and idx - 1
        # We want to minimize |a_sum + b_sum| != 0
        # So we look for b_sum closest to -a_sum
        
        candidates = []
        if idx < len(sorted_sums_b):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for c_idx in candidates:
            b_sum = sorted_sums_b[c_idx]
            total = a_sum + b_sum
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_val:
                    best_val = abs_total
                    best_mask_a = masks_a[i]
                    best_mask_b = sorted_masks_b[c_idx]
    
    # Reconstruct coefficients
    coeffs = [0] * n
    
    if best_mask_a != -1:
        # Decode mask_a
        for i in range(mid):
            digit = (best_mask_a // (3 ** i)) % 3
            coeffs[i] = digit - 1
            
        # Decode mask_b
        for i in range(mid):
            digit = (best_mask_b // (3 ** i)) % 3
            coeffs[mid + i] = digit - 1
    else:
        # Fallback
        coeffs[29] = 1
        coeffs[0] = 1
        s = sum(c * W[i] for i, c in enumerate(coeffs))
        if s == 0:
            coeffs[29] = -1

    return coeffs
