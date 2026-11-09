import math
import time
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    n = 30
    half = 15

    # Split into two halves: indices 0..14 and 15..29
    # Each coefficient c[i] multiplies 1/(i+1)
    # Half 1: i=0..14, denominators 1..15
    # Half 2: i=15..29, denominators 16..30

    # Generate all 3^15 sums for each half using exact fractions
    # But fractions are slow, so let's use a common denominator approach
    
    # LCM of 1..30
    def lcm(a, b):
        return a // math.gcd(a, b) * b

    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)

    # LCM is about 232908956280000
    # This is manageable as int64? No, it's about 2.3e14, fits in int64 (max ~9.2e18)
    
    # Scaled weights: weight[i] = L // (i+1)
    weights = [L // (i + 1) for i in range(n)]
    w1 = weights[:half]
    w2 = weights[half:]

    # Generate all sums for half 1
    def gen_sums(w):
        sums = np.array([0], dtype=np.int64)
        for wi in w:
            sums = np.concatenate((sums - wi, sums, sums + wi))
        return sums

    sums1 = gen_sums(w1)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    sums2 = gen_sums(w2)

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)

    # For each sum in sums1, find the sum in sums2 that makes |s1+s2| minimal and nonzero
    targets = -sums1
    indices = np.searchsorted(sums2_sorted, targets)

    len2 = len(sums2_sorted)
    max_int64 = np.iinfo(np.int64).max

    # Candidate 1: sums2_sorted[indices] (first element >= target)
    idx1 = indices
    mask1 = (idx1 >= 0) & (idx1 < len2)
    cand1_s2 = np.zeros(len(sums1), dtype=np.int64)
    cand1_s2[mask1] = sums2_sorted[idx1[mask1]]
    total1 = sums1 + cand1_s2
    valid1 = mask1 & (total1 != 0)
    abs1 = np.where(valid1, np.abs(total1), max_int64)

    # Candidate 2: sums2_sorted[indices-1] (last element < target)
    idx2 = indices - 1
    mask2 = (idx2 >= 0) & (idx2 < len2)
    cand2_s2 = np.zeros(len(sums1), dtype=np.int64)
    cand2_s2[mask2] = sums2_sorted[idx2[mask2]]
    total2 = sums1 + cand2_s2
    valid2 = mask2 & (total2 != 0)
    abs2 = np.where(valid2, np.abs(total2), max_int64)

    # Find global minimum
    min_abs1 = np.min(abs1)
    min_abs2 = np.min(abs2)

    if min_abs1 < min_abs2:
        best_val = min_abs1
        idx_min = np.argmin(abs1)
        best_s1 = sums1[idx_min]
        best_s2 = cand1_s2[idx_min]
    else:
        best_val = min_abs2
        idx_min = np.argmin(abs2)
        best_s1 = sums1[idx_min]
        best_s2 = cand2_s2[idx_min]

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Backtrack to find coefficients for best_s1 and best_s2
    def find_coeffs(w, target):
        coeffs = [0] * len(w)
        def dfs(pos, current):
            if pos == len(w):
                return current == target
            wi = w[pos]
            coeffs[pos] = -1
            if dfs(pos + 1, current - wi):
                return True
            coeffs[pos] = 0
            if dfs(pos + 1, current):
                return True
            coeffs[pos] = 1
            if dfs(pos + 1, current + wi):
                return True
            coeffs[pos] = 0
            return False
        if dfs(0, 0):
            return coeffs
        return None

    c1 = find_coeffs(w1, best_s1)
    c2 = find_coeffs(w2, best_s2)

    if c1 is None or c2 is None:
        return [0] * 28 + [1, -1]

    coeffs = c1 + c2

    # Final verification
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(coeffs))
    if total_frac == 0:
        return [0] * 28 + [1, -1]

    return coeffs
