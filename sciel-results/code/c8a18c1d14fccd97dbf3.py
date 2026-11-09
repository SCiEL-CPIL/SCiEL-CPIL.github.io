import math
import time
import bisect
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns exactly 30 integer coefficients from {-1, 0, 1} such that
    sum(coeff[i-1]/i for i in 1..30) is nonzero and has minimal absolute value.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    n = 30
    half = 15

    # Calculate LCM of 1 to 30
    def lcm(a, b):
        return a // math.gcd(a, b) * b

    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)

    # Scaled weights: weight[i] = L // (i+1)
    weights = [L // (i + 1) for i in range(n)]
    w1 = weights[:half]
    w2 = weights[half:]

    def gen_sums(w):
        """Generate all 3^len(w) possible sums using numpy concatenation."""
        sums = np.array([0], dtype=np.int64)
        for wi in w:
            sums = np.concatenate((sums - wi, sums, sums + wi))
        return sums

    # Generate all sums for both halves
    sums1 = gen_sums(w1)
    sums2 = gen_sums(w2)

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)

    # Vectorized search for minimum |s1 + s2| where s1+s2 != 0
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
        """DFS to find coefficients in {-1,0,1} such that sum(c[i]*w[i]) == target."""
        coeffs = [0] * len(w)
        def dfs(pos, current):
            if pos == len(w):
                return current == target
            wi = w[pos]
            # Try -1, 0, 1
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
        # Fallback
        return [0] * 28 + [1, -1]

    coeffs = c1 + c2

    # Final verification
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(coeffs))
    if total_frac == 0:
        return [0] * 28 + [1, -1]

    return coeffs

# EVOLVE-BLOCK-END
