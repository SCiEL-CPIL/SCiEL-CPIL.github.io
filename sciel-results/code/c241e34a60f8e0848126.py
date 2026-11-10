"""Strategy: Meet-in-the-Middle with NumPy Vectorization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = 30
    # We want to minimize |sum c_i / i| != 0
    # c_i in {-1, 0, 1}
    
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # Precompute weights
    weights = [L // (i + 1) for i in range(n)]
    
    n1 = n // 2
    n2 = n - n1
    
    # Generate sums and codes for first half
    # Code is a 2*n1 bit integer, 2 bits per coefficient (0:0, 1:1, 2:-1)
    sums1 = np.array([0], dtype=np.int64)
    codes1 = np.array([0], dtype=np.int64)
    
    for i in range(n1):
        w = weights[i]
        # Expand arrays
        # New sums: s, s+w, s-w
        # New codes: c, c+1<<(2*i), c+2<<(2*i)
        
        s_new = np.empty(len(sums1) * 3, dtype=np.int64)
        c_new = np.empty(len(codes1) * 3, dtype=np.int64)
        
        s_new[0::3] = sums1
        s_new[1::3] = sums1 + w
        s_new[2::3] = sums1 - w
        
        c_new[0::3] = codes1
        c_new[1::3] = codes1 + (1 << (2 * i))
        c_new[2::3] = codes1 + (2 << (2 * i))
        
        sums1 = s_new
        codes1 = c_new
        
    # Generate sums and codes for second half
    sums2 = np.array([0], dtype=np.int64)
    codes2 = np.array([0], dtype=np.int64)
    
    for i in range(n2):
        w = weights[n1 + i]
        
        s_new = np.empty(len(sums2) * 3, dtype=np.int64)
        c_new = np.empty(len(codes2) * 3, dtype=np.int64)
        
        s_new[0::3] = sums2
        s_new[1::3] = sums2 + w
        s_new[2::3] = sums2 - w
        
        c_new[0::3] = codes2
        c_new[1::3] = codes2 + (1 << (2 * i))
        c_new[2::3] = codes2 + (2 << (2 * i))
        
        sums2 = s_new
        codes2 = c_new
        
    # Sort second half by sum
    sort_idx = np.argsort(sums2)
    sums2_sorted = sums2[sort_idx]
    codes2_sorted = codes2[sort_idx]
    
    best_val = None
    best_coeffs = None
    
    # Vectorized search
    # For each s1 in sums1, we want s2 in sums2_sorted closest to -s1
    # target = -sums1
    targets = -sums1
    
    # Find insertion points
    positions = np.searchsorted(sums2_sorted, targets)
    
    # Candidates: positions and positions - 1
    # Clip to valid range
    pos_left = np.clip(positions - 1, 0, len(sums2_sorted) - 1)
    pos_right = np.clip(positions, 0, len(sums2_sorted) - 1)
    
    # Get candidate sums and codes
    s2_left = sums2_sorted[pos_left]
    s2_right = sums2_sorted[pos_right]
    
    c2_left = codes2_sorted[pos_left]
    c2_right = codes2_sorted[pos_right]
    
    # Calculate total sums
    total_left = sums1 + s2_left
    total_right = sums1 + s2_right
    
    # Calculate absolute sums
    abs_left = np.abs(total_left)
    abs_right = np.abs(total_right)
    
    # Mask out zero sums
    mask_left = total_left != 0
    mask_right = total_right != 0
    
    # Set invalid ones to infinity
    abs_left[~mask_left] = np.iinfo(np.int64).max
    abs_right[~mask_right] = np.iinfo(np.int64).max
    
    # Find global minimum
    min_abs_left = np.min(abs_left)
    min_abs_right = np.min(abs_right)
    
    if min_abs_left < min_abs_right:
        best_idx = np.argmin(abs_left)
        best_val = min_abs_left
        s2_best = s2_left[best_idx]
        c2_best = c2_left[best_idx]
    else:
        best_idx = np.argmin(abs_right)
        best_val = min_abs_right
        s2_best = s2_right[best_idx]
        c2_best = c2_right[best_idx]
        
    # Reconstruct coefficients
    if best_val is not None and best_val != np.iinfo(np.int64).max:
        c1_best = codes1[best_idx]
        coeffs = [0] * n
        
        # Decode code1
        c = c1_best
        for i in range(n1):
            val = (c >> (2 * i)) & 3
            if val == 1:
                coeffs[i] = 1
            elif val == 2:
                coeffs[i] = -1
                
        # Decode code2
        c = c2_best
        for i in range(n2):
            val = (c >> (2 * i)) & 3
            if val == 1:
                coeffs[n1 + i] = 1
            elif val == 2:
                coeffs[n1 + i] = -1
                
        return coeffs
        
    # Fallback
    return [0] * 28 + [1, -1]
# EVOLVE-BLOCK-END
