"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction
from math import isqrt
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
    # sigma[n] = sum of all positive divisors of n
    # For each i from 1 to N, add i to sigma[j] for all j that are multiples of i
    # This is O(N log N) but vectorized with NumPy
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        if time_left() < 15:
            break
        sigma[i::i] += i
    
    if time_left() < 10:
        return [11, 6]
    
    # Step 2: Find the best pair (a, b) with sigma(a) = sigma(b) and a/b ≈ 3/2
    # Strategy: iterate over small error values k = 2a - 3b
    # For each k, find b such that a = (3b + k) / 2 is an integer
    # This means 3b + k must be even, so k and b must have the same parity
    # a = (3b + k) // 2, and we need 1 <= a <= N, 1 <= b <= N, a != b
    # sigma(a) = sigma(b)
    # Minimize |k| / (2b)
    
    best_err = None
    best_pair = (11, 6)
    
    # Check the seed first
    if sigma[11] == sigma[6]:
        best_err = abs(Fraction(11, 6) - TARGET)
    
    # For each k starting from small values, search for b
    # k = 2a - 3b, so a = (3b + k) / 2
    # For a to be integer: 3b + k must be even
    # If k is even, b must be even (3b is odd when b is odd, so 3b+k is odd+even=odd, not even)
    # Wait: 3b + k even means 3b and k have same parity
    # 3b has same parity as b, so b and k must have same parity
    
    # For a given k, we need to find b such that:
    # 1. b has same parity as k
    # 2. 1 <= (3b + k) // 2 <= N
    # 3. 1 <= b <= N
    # 4. (3b + k) // 2 != b (i.e., a != b, which means k != 0)
    # 5. sigma((3b + k) // 2) == sigma(b)
    
    # From constraint 2: 2 <= 3b + k <= 2N, so (2-k)/3 <= b <= (2N-k)/3
    # Also b >= 1 and b <= N
    
    # Let's iterate over k in order of |k|, and for each k, iterate over b in a smart way
    # To minimize |k|/(2b), for a fixed k, we want large b
    # So for each k, start from the largest valid b and work down
    
    # But this could be very slow. Let's think differently.
    
    # Alternative: for each b, the target a is near 3b/2
    # Let's check a = floor(3b/2) and a = ceil(3b/2) for each b
    # This is O(N) sigma lookups, which is fast
    
    # Vectorized approach:
    b_arr = np.arange(1, N + 1, dtype=np.int64)
    
    # a = 3b // 2 (floor)
    a1 = (3 * b_arr) // 2
    valid1 = (a1 >= 1) & (a1 <= N) & (a1 != b_arr)
    a1_valid = a1[valid1]
    b1_valid = b_arr[valid1]
    
    sigma_a1 = sigma[a1_valid]
    sigma_b1 = sigma[b1_valid]
    matches1 = sigma_a1 == sigma_b1
    
    # a = (3b + 1) // 2 (ceil when 3b is odd, i.e., b is odd)
    a2 = (3 * b_arr + 1) // 2
    valid2 = (a2 >= 1) & (a2 <= N) & (a2 != b_arr)
    a2_valid = a2[valid2]
    b2_valid = b_arr[valid2]
    
    sigma_a2 = sigma[a2_valid]
    sigma_b2 = sigma[b2_valid]
    matches2 = sigma_a2 == sigma_b2
    
    # Combine all matches
    all_a = np.concatenate([a1_valid[matches1], a2_valid[matches2]])
    all_b = np.concatenate([b1_valid[matches1], b2_valid[matches2]])
    
    if len(all_a) > 0:
        # Compute error |a/b - 3/2| = |2a - 3b| / (2b)
        num = np.abs(2 * all_a - 3 * all_b)
        den = 2 * all_b
        ratios = num.astype(np.float64) / den.astype(np.float64)
        min_idx = np.argmin(ratios)
        
        a_best = int(all_a[min_idx])
        b_best = int(all_b[min_idx])
        
        if a_best != b_best and 1 <= a_best <= N and 1 <= b_best <= N:
            if sigma[a_best] == sigma[b_best]:
                err = abs(Fraction(a_best, b_best) - TARGET)
                if best_err is None or err < best_err:
                    best_err = err
                    best_pair = (a_best, b_best)
    
    # Step 3: Local polish - for the best b found, check neighbors of 3b/2
    if best_pair[1] > 0 and time_left() > 5:
        b_center = best_pair[1]
        a_center = 3 * b_center // 2
        for da in range(-100, 101):
            if time_left() < 5:
                break
            a_cand = a_center + da
            if 1 <= a_cand <= N and a_cand != b_center:
                if sigma[a_cand] == sigma[b_center]:
                    err = abs(Fraction(a_cand, b_center) - TARGET)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_pair = (a_cand, b_center)
    
    # Step 4: Also try searching over small k values directly
    if time_left() > 10:
        for k in range(1, 200):
            if time_left() < 10:
                break
            # For this k, find b such that a = (3b + k) / 2 is integer
            # b must have same parity as k
            # b range: max(1, ceil((2-k)/3)) to min(N, floor((2N-k)/3))
            b_min = max(1, (2 - k + 2) // 3)  # ceil((2-k)/3)
            if k % 2 == 1:
                b_min = max(b_min, 1)
                if b_min % 2 == 0:
                    b_min += 1
            else:
                b_min = max(b_min, 2)
                if b_min % 2 == 1:
                    b_min += 1
            b_max = min(N, (2 * N - k) // 3)
            if k % 2 == 1:
                if b_max % 2 == 0:
                    b_max -= 1
            else:
                if b_max % 2 == 1:
                    b_max -= 1
            
            if b_min > b_max:
                continue
            
            # For this k, we want to maximize b to minimize k/(2b)
            # Start from b_max and go down
            # But we can only check a limited number of b values
            # Let's check b_max, b_max-2, b_max-4, ... (step 2 to maintain parity)
            # Limit to first 1000 candidates per k
            
            for b_cand in range(b_max, b_min - 1, -2):
                if time_left() < 10:
                    break
                a_cand = (3 * b_cand + k) // 2
                if a_cand < 1 or a_cand > N or a_cand == b_cand:
                    continue
                if sigma[a_cand] == sigma[b_cand]:
                    err = abs(Fraction(a_cand, b_cand) - TARGET)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_pair = (a_cand, b_cand)
    
    # Verify the result
    a_final, b_final = best_pair
    if not (1 <= a_final <= N and 1 <= b_final <= N):
        return [11, 6]
    if sigma[a_final] != sigma[b_final]:
        return [11, 6]
    
    err_final = abs(Fraction(a_final, b_final) - TARGET)
    if err_final == 0:
        return [11, 6]
    
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
