import numpy as np
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    n = 30
    half = 15
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute scaled terms
    terms = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    size = 3 ** half
    
    def gen_sums_with_idx(terms_part):
        sums = np.array([0], dtype=np.int64)
        idx = np.array([0], dtype=np.int64)
        
        powers = 3 ** np.arange(len(terms_part))
        
        for i, t in enumerate(terms_part):
            curr_size = len(sums)
            new_sums = np.empty(curr_size * 3, dtype=np.int64)
            new_idx = np.empty(curr_size * 3, dtype=np.int64)
            
            # Digit 0 (coeff -1)
            new_sums[0:curr_size] = sums - t
            new_idx[0:curr_size] = idx + 0 * powers[i]
            
            # Digit 1 (coeff 0)
            new_sums[curr_size:2*curr_size] = sums
            new_idx[curr_size:2*curr_size] = idx + 1 * powers[i]
            
            # Digit 2 (coeff 1)
            new_sums[2*curr_size:3*curr_size] = sums + t
            new_idx[2*curr_size:3*curr_size] = idx + 2 * powers[i]
            
            sums = new_sums
            idx = new_idx
            
            if time.time() > deadline:
                # Fallback: return a simple nonzero sum
                # e.g. 1/1 - 1/1 is 0, so 1/1 - 1/2 = 1/2
                # Coeffs: [1, -1, 0, ...]
                return None, None
        
        return sums, idx

    # Generate for A (terms 0-14)
    sums_A, idx_A = gen_sums_with_idx(terms[0:15])
    
    if sums_A is None:
        return [1, -1] + [0]*28

    # Generate for B (terms 15-29)
    sums_B, idx_B = gen_sums_with_idx(terms[15:30])
    
    if sums_B is None:
        return [1, -1] + [0]*28

    if time.time() > deadline:
        return [1, -1] + [0]*28

    # Sort sums_B and reorder idx_B
    sort_idx = np.argsort(sums_B)
    sums_B_sorted = sums_B[sort_idx]
    idx_B_sorted = idx_B[sort_idx]
    
    # Search for best pair
    # We want min |s_A + s_B| != 0
    
    targets = -sums_A
    pos = np.searchsorted(sums_B_sorted, targets)
    
    # Candidate 1: sums_B_sorted[pos]
    pos1 = np.clip(pos, 0, len(sums_B_sorted) - 1)
    cand_sums_B_1 = sums_B_sorted[pos1]
    total_1 = sums_A + cand_sums_B_1
    
    # Candidate 2: sums_B_sorted[pos-1]
    pos2 = np.clip(pos - 1, 0, len(sums_B_sorted) - 1)
    cand_sums_B_2 = sums_B_sorted[pos2]
    total_2 = sums_A + cand_sums_B_2
    
    # Combine candidates
    # We have 2 * size candidates
    # total_1 corresponds to idx_A = arange(size), idx_B = pos1
    # total_2 corresponds to idx_A = arange(size), idx_B = pos2
    
    idx_A_arr = np.arange(len(sums_A), dtype=np.int64)
    
    # Create arrays for all candidates
    # To save memory, we don't necessarily need to concatenate if we can find min in two passes,
    # but concatenating 2 * 14M int64s is ~224MB, which is acceptable.
    
    all_totals = np.concatenate([total_1, total_2])
    all_idx_A = np.concatenate([idx_A_arr, idx_A_arr])
    all_idx_B = np.concatenate([pos1, pos2])
    
    # Filter out zeros
    mask = all_totals != 0
    if not np.any(mask):
        # This should not happen, but fallback
        return [1, -1] + [0]*28
        
    # Find min absolute value
    # We only look at non-zeros
    abs_totals = np.abs(all_totals)
    abs_totals[~mask] = np.iinfo(np.int64).max
    
    min_idx = np.argmin(abs_totals)
    
    best_total = all_totals[min_idx]
    best_idx_A = all_idx_A[min_idx]
    best_idx_B = all_idx_B[min_idx]
    
    # Recover coefficients
    coeff_idx_A = idx_A[best_idx_A]
    coeff_idx_B = idx_B_sorted[best_idx_B]
    
    coeffs = np.zeros(30, dtype=int)
    
    powers = 3 ** np.arange(15)
    
    # Decode A
    digits_A = (coeff_idx_A // powers) % 3
    # Map 0->-1, 1->0, 2->1
    coeffs[0:15] = np.where(digits_A == 0, -1, np.where(digits_A == 1, 0, 1))
    
    # Decode B
    digits_B = (coeff_idx_B // powers) % 3
    coeffs[15:30] = np.where(digits_B == 0, -1, np.where(digits_B == 1, 0, 1))
    
    # Verify sum is nonzero
    # s = sum(coeffs[i] / (i+1))
    # if s == 0: fallback
    
    # Return as Python integers
    return [int(x) for x in coeffs]
