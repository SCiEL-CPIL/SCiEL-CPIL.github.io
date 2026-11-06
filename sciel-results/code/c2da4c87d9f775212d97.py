import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    """
    Find 30 coefficients in {-1, 0, 1} such that sum(c[i]/(i+1)) is nonzero
    and has minimal absolute value.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute weights
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        """
        Generate all possible sums for a half using coefficients in {-1, 0, 1}.
        Returns a numpy array of sums.
        The order of sums corresponds to base-3 representation of the index,
        where the least significant digit corresponds to the first weight,
        and the most significant digit corresponds to the last weight.
        """
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    # Generate sums for both halves
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 and keep track of original indices
    sorted_sums2, sorted_indices2 = np.sort(sums2), np.argsort(sums2)
    
    best_abs_S = np.iinfo(np.int64).max
    best_idx1 = -1
    best_idx2_sorted = -1
    
    # Vectorized search
    targets = -sums1
    indices = np.searchsorted(sorted_sums2, targets)
    
    len_s2 = len(sorted_sums2)
    
    # Right candidates (first element >= target)
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sorted_sums2[idx_r]
    total_r = sums1 + s2_r
    
    # Left candidates (last element < target)
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sorted_sums2[idx_l]
    total_l = sums1 + s2_l
    
    # We want min |total| for total != 0
    abs_total_r = np.abs(total_r)
    abs_total_l = np.abs(total_l)
    
    # Mask out zeros
    abs_total_r[total_r == 0] = np.iinfo(np.int64).max
    abs_total_l[total_l == 0] = np.iinfo(np.int64).max
    
    # Find the global minimum among the two candidate sets
    min_abs_r = np.min(abs_total_r)
    min_abs_l = np.min(abs_total_l)
    
    if min_abs_r < min_abs_l:
        best_abs_S = min_abs_r
        best_idx1 = np.argmin(abs_total_r)
        best_idx2_sorted = idx_r[best_idx1]
    else:
        best_abs_S = min_abs_l
        best_idx1 = np.argmin(abs_total_l)
        best_idx2_sorted = idx_l[best_idx1]
        
    # Check if we found a valid solution
    if best_abs_S == np.iinfo(np.int64).max:
        # This implies all candidates were zero, which is impossible for non-trivial weights.
        # But just in case, do a local scan around the best zero candidate.
        # Find the s1 that produced a zero sum (or closest to zero)
        # Actually, if min_abs is inf, it means total_r and total_l were all 0?
        # That would mean for every s1, the closest s2 is -s1.
        # This is unlikely.
        # Let's just pick the first s1 and find the nearest non-zero s2.
        s1_val = sums1[0]
        target = -s1_val
        pos = np.searchsorted(sorted_sums2, target)
        # Scan neighbors
        for delta in range(0, 100):
            for d in [delta, -delta]:
                if delta == 0 and d < 0: continue
                idx = pos + d
                if 0 <= idx < len_s2:
                    s2_val = sorted_sums2[idx]
                    total = s1_val + s2_val
                    if total != 0:
                        best_abs_S = abs(total)
                        best_idx1 = 0
                        best_idx2_sorted = idx
                        break
            if best_abs_S != np.iinfo(np.int64).max:
                break

    # Reconstruct coefficients
    # orig_idx1 is best_idx1
    # orig_idx2 is sorted_indices2[best_idx2_sorted]
    
    orig_idx1 = best_idx1
    orig_idx2 = sorted_indices2[best_idx2_sorted]
    
    def decode_coeffs(idx, n_weights):
        """
        Decode base-3 index to coefficients.
        Least significant digit corresponds to weight[0].
        """
        coeffs = np.zeros(n_weights, dtype=np.int64)
        for i in range(n_weights):
            digit = idx % 3
            coeffs[i] = digit - 1
            idx //= 3
        return coeffs
    
    c1 = decode_coeffs(orig_idx1, mid)
    c2 = decode_coeffs(orig_idx2, n - mid)
    
    final_coeffs = np.concatenate([c1, c2])
    
    # Verify
    # val = sum(c[i] / (i+1) for i in range(30))
    # assert val != 0
    
    return list(final_coeffs)
