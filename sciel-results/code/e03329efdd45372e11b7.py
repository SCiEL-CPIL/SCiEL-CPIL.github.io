import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute weights
    # w[i] = L / (i+1) for i in 0..29
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums_with_coeffs(weights_half):
        """
        Generate all possible sums and their coefficient patterns for a half.
        Returns (sums_array, coeffs_array) where coeffs_array has shape (num_sums, len(weights_half))
        """
        n_half = len(weights_half)
        sums = np.array([0], dtype=np.int64)
        coeffs = np.zeros((1, n_half), dtype=np.int8)
        
        for i, w in enumerate(weights_half):
            # For each existing sum, create three new sums: s-w, s, s+w
            new_sums = np.concatenate([sums - w, sums, sums + w])
            
            # Update coefficients
            # For each existing coeff pattern, create three new patterns
            # The new coefficient for position i is -1, 0, or 1
            n_existing = len(sums)
            new_coeffs = np.zeros((3 * n_existing, n_half), dtype=np.int8)
            new_coeffs[:n_existing, :] = coeffs
            new_coeffs[:n_existing, i] = -1
            new_coeffs[n_existing:2*n_existing, :] = coeffs
            new_coeffs[n_existing:2*n_existing, i] = 0
            new_coeffs[2*n_existing:, :] = coeffs
            new_coeffs[2*n_existing:, i] = 1
            
            sums = new_sums
            coeffs = new_coeffs
        
        return sums, coeffs

    # Generate sums and coefficients for both halves
    # 3^15 is ~14.3 million
    sums1, coeffs1_arr = generate_sums_with_coeffs(w1)
    sums2, coeffs2_arr = generate_sums_with_coeffs(w2)
    
    # Sort sums2 and keep track of original indices
    sort_idx = np.argsort(sums2)
    sums2_sorted = sums2[sort_idx]
    coeffs2_sorted = coeffs2_arr[sort_idx]
    
    len_s2 = len(sums2_sorted)
    
    # We want to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # Vectorized search
    targets = -sums1
    
    # Find insertion points
    indices = np.searchsorted(sums2_sorted, targets)
    
    # Check multiple neighbors: indices-2 to indices+2
    # Create candidate index arrays
    candidates = []
    for offset in [-2, -1, 0, 1, 2]:
        idx = np.clip(indices + offset, 0, len_s2 - 1)
        s2_vals = sums2_sorted[idx]
        total_vals = sums1 + s2_vals
        candidates.append(total_vals)
    
    # Stack all candidates
    all_totals = np.vstack(candidates)  # shape (5, len_sums1)
    
    # Mask zeros
    mask = all_totals != 0
    
    # Set zeros to max value
    abs_totals = np.abs(all_totals).astype(np.int64)
    abs_totals[~mask] = np.iinfo(np.int64).max
    
    # Find global minimum
    flat_idx = np.argmin(abs_totals)
    
    # Recover which candidate and which sums1 index
    n_sums1 = len(sums1)
    row = flat_idx // n_sums1
    col = flat_idx % n_sums1
    
    s1_val = sums1[col]
    coeffs1 = coeffs1_arr[col]
    
    # Get the corresponding s2 index
    offset = [-2, -1, 0, 1, 2][row]
    idx = np.clip(indices[col] + offset, 0, len_s2 - 1)
    s2_val = sums2_sorted[idx]
    coeffs2 = coeffs2_sorted[idx]
    
    # Convert to Python int lists
    result = [int(c) for c in coeffs1] + [int(c) for c in coeffs2]
    
    return result
