"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction
import numpy as np

MAX_N = 10_000_000
TARGET = Fraction(3, 2)
DEADLINE_SECONDS = 80  # safety margin before 90s kill

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE_SECONDS - (time.time() - t0)
    
    N = MAX_N
    
    # Step 1: Compute sigma(n) for all n up to N using NumPy sieve
    # sigma[n] = sum of all divisors of n
    # Method: for each i, add i to all multiples of i
    sigma_np = np.zeros(N + 1, dtype=np.int64)
    
    # This is O(N log N) but NumPy vectorizes the inner loop
    # Total operations: N * H(N) ≈ 10M * 16.7 ≈ 167M
    for i in range(1, N + 1):
        if time_left() < 15:
            break
        sigma_np[i::i] += i
    
    if time_left() < 5:
        # Fallback: return a known valid pair
        # sigma(11) = 1 + 11 = 12, sigma(6) = 1 + 2 + 3 + 6 = 12
        return [11, 6]
    
    # Step 2: Find pairs (a, b) with sigma(a) = sigma(b) and a/b close to 3/2
    # For each b, the ideal a is 3b/2. We check a near 3b/2.
    
    # Create array of b values
    b_arr = np.arange(1, N + 1, dtype=np.int64)
    
    # Target a for each b: a_target = round(3b/2)
    # But we need to check a window around 3b/2
    # To be efficient, first check exact a = 3b//2 and a = (3b+1)//2
    
    # Check a = 3b // 2
    a1 = (3 * b_arr) // 2
    valid1 = (a1 >= 1) & (a1 <= N) & (a1 != b_arr)
    
    if np.any(valid1):
        a1v = a1[valid1]
        b1v = b_arr[valid1]
        match1 = sigma_np[a1v] == sigma_np[b1v]
        if np.any(match1):
            a1m = a1v[match1]
            b1m = b1v[match1]
            # Error: |a/b - 3/2| = |2a - 3b| / (2b)
            err_num1 = np.abs(2 * a1m - 3 * b1m)
            err_den1 = 2 * b1m
            # Find minimum
            # Use float for comparison, then verify with Fraction
            ratios1 = err_num1.astype(np.float64) / err_den1.astype(np.float64)
            min_idx1 = np.argmin(ratios1)
            a_best1 = int(a1m[min_idx1])
            b_best1 = int(b1m[min_idx1])
            err_best1 = abs(Fraction(a_best1, b_best1) - TARGET)
        else:
            err_best1 = None
            a_best1 = b_best1 = None
    else:
        err_best1 = None
        a_best1 = b_best1 = None
    
    # Check a = (3b + 1) // 2 (the other rounding)
    a2 = (3 * b_arr + 1) // 2
    valid2 = (a2 >= 1) & (a2 <= N) & (a2 != b_arr)
    
    if np.any(valid2):
        a2v = a2[valid2]
        b2v = b_arr[valid2]
        match2 = sigma_np[a2v] == sigma_np[b2v]
        if np.any(match2):
            a2m = a2v[match2]
            b2m = b2v[match2]
            err_num2 = np.abs(2 * a2m - 3 * b2m)
            err_den2 = 2 * b2m
            ratios2 = err_num2.astype(np.float64) / err_den2.astype(np.float64)
            min_idx2 = np.argmin(ratios2)
            a_best2 = int(a2m[min_idx2])
            b_best2 = int(b2m[min_idx2])
            err_best2 = abs(Fraction(a_best2, b_best2) - TARGET)
        else:
            err_best2 = None
            a_best2 = b_best2 = None
    else:
        err_best2 = None
        a_best2 = b_best2 = None
    
    # Choose the best of the two
    best_err = None
    best_pair = (11, 6)  # fallback
    
    if err_best1 is not None:
        best_err = err_best1
        best_pair = (a_best1, b_best1)
    if err_best2 is not None:
        if best_err is None or err_best2 < best_err:
            best_err = err_best2
            best_pair = (a_best2, b_best2)
    
    # Step 3: Refine by checking neighbors of the best b
    if best_pair[1] > 0 and time_left() > 2:
        b_center = best_pair[1]
        a_center = 3 * b_center // 2
        # Check a window of +/- 100 around a_center
        for da in range(-100, 101):
            a_cand = a_center + da
            if 1 <= a_cand <= N and a_cand != b_center:
                if sigma_np[a_cand] == sigma_np[b_center]:
                    err = abs(Fraction(a_cand, b_center) - TARGET)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_pair = (a_cand, b_center)
    
    # Also try refining b: check b in a window around the best b
    if time_left() > 3:
        b_best_curr = best_pair[1]
        for db in range(-50, 51):
            b_cand = b_best_curr + db
            if 1 <= b_cand <= N and b_cand != best_pair[0]:
                a_target = 3 * b_cand // 2
                for da in range(-20, 21):
                    a_cand = a_target + da
                    if 1 <= a_cand <= N and a_cand != b_cand:
                        if sigma_np[a_cand] == sigma_np[b_cand]:
                            err = abs(Fraction(a_cand, b_cand) - TARGET)
                            if best_err is None or err < best_err:
                                best_err = err
                                best_pair = (a_cand, b_cand)
    
    # Verify the final result
    a_final, b_final = best_pair
    if not (1 <= a_final <= N and 1 <= b_final <= N):
        return [11, 6]
    if a_final == b_final:
        return [11, 6]
    if sigma_np[a_final] != sigma_np[b_final]:
        return [11, 6]
    
    err_final = abs(Fraction(a_final, b_final) - TARGET)
    if err_final == 0:
        return [11, 6]
    
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
