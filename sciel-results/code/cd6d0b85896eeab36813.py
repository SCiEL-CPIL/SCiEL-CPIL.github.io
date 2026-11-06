import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1), 1)
    
    # Weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    def generate_sums_and_coeffs(weights_subset):
        """
        Generates all possible sums and their corresponding coefficient vectors.
        Returns:
            sums: numpy array of shape (3^k,)
            coeffs: numpy array of shape (3^k, k) with values in {-1, 0, 1}
        """
        k = len(weights_subset)
        sums = np.array([0], dtype=np.int64)
        coeffs = np.zeros((1, k), dtype=np.int8)
        
        for i, w in enumerate(weights_subset):
            sums_neg = sums - w
            sums_pos = sums + w
            
            new_sums = np.concatenate([sums_neg, sums, sums_pos])
            
            c_neg = coeffs.copy()
            c_pos = coeffs.copy()
            
            c_neg[:, i] = -1
            c_pos[:, i] = 1
            
            new_coeffs = np.concatenate([c_neg, coeffs, c_pos], axis=0)
            
            sums = new_sums
            coeffs = new_coeffs
            
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded during generation")
                
        return sums, coeffs

    try:
        sums1, coeffs1 = generate_sums_and_coeffs(w1)
        sums2, coeffs2 = generate_sums_and_coeffs(w2)
    except TimeoutError:
        # Fallback to a greedy heuristic
        result = [0] * n
        current_sum = 0.0
        for i in range(n):
            best_c = 0
            best_abs = float('inf')
            for c in [-1, 0, 1]:
                new_sum = current_sum + c / (i + 1)
                a = abs(new_sum)
                if a > 1e-15 and a < best_abs:
                    best_abs = a
                    best_c = c
            result[i] = best_c
            current_sum += best_c / (i + 1)
        total = sum(result[i] / (i + 1) for i in range(n))
        if abs(total) < 1e-15:
            result[-1] = 1 if result[-1] != 1 else -1
        return result

    # Sort sums2
    sort_indices = np.argsort(sums2)
    sorted_sums2 = sums2[sort_indices]
    
    # Vectorized search
    targets = -sums1
    indices = np.searchsorted(sorted_sums2, targets)
    
    idx1 = np.clip(indices, 0, len(sorted_sums2) - 1)
    idx2 = np.clip(indices - 1, 0, len(sorted_sums2) - 1)
    
    cand_sums1 = sorted_sums2[idx1]
    cand_sums2 = sorted_sums2[idx2]
    
    total1 = sums1 + cand_sums1
    total2 = sums1 + cand_sums2
    
    max_int = np.iinfo(np.int64).max
    abs_total1 = np.where(total1 == 0, max_int, np.abs(total1))
    abs_total2 = np.where(total2 == 0, max_int, np.abs(total2))
    
    min1 = np.min(abs_total1)
    min2 = np.min(abs_total2)
    
    if min1 < min2:
        best_val = min1
        best_i1 = np.argmin(abs_total1)
        best_i2_sorted = idx1[best_i1]
    else:
        best_val = min2
        best_i1 = np.argmin(abs_total2)
        best_i2_sorted = idx2[best_i1]
    
    if best_val == max_int:
        return [0] * 28 + [1, -1]
    
    orig_idx2 = sort_indices[best_i2_sorted]
    
    c1 = coeffs1[best_i1]
    c2 = coeffs2[orig_idx2]
    
    result = np.concatenate([c1, c2]).astype(int).tolist()
    
    return result
