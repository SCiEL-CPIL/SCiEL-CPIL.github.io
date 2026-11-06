import time
import random
from math import gcd
from functools import reduce
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 30

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, n + 1))

    # W[i] = L // (i+1), so sum(c[i]/(i+1)) = (1/L) * sum(c[i]*W[i])
    # We want to minimize |sum(c[i]*W[i])| subject to it being nonzero
    W = [L // (i + 1) for i in range(n)]

    best_N = None
    best_coeffs = None
    best_abs_N = float('inf')

    def calc_N(coeffs):
        s = 0
        for i in range(n):
            c = coeffs[i]
            if c != 0:
                s += c * W[i]
        return s

    # Meet-in-the-middle approach for exact global optimum
    # Split into two halves of 15 each
    half = n // 2  # 15
    half2 = n - half  # 15

    # Generate all 3^15 = 14348901 combinations for each half
    # This is feasible with vectorized operations

    import numpy as np

    # Generate all coefficient combinations for first half
    # Each coefficient is -1, 0, or 1
    # Use base-3 representation: 0->-1, 1->0, 2->1
    def gen_all_coeffs(count):
        """Generate all 3^count coefficient vectors as a numpy array of shape (3^count, count)"""
        total = 3 ** count
        # Use efficient generation
        coeffs = np.zeros((total, count), dtype=np.int8)
        for i in range(total):
            x = i
            for j in range(count):
                digit = x % 3
                x //= 3
                if digit == 0:
                    coeffs[i, j] = -1
                elif digit == 1:
                    coeffs[i, j] = 0
                else:
                    coeffs[i, j] = 1
        return coeffs

    # This might be too slow for 3^15. Let's use a smarter approach.
    # Instead, we'll use meet-in-the-middle with sorting.

    # Generate sums for first half
    def gen_sums_half(indices):
        """Generate all possible weighted sums for a subset of indices."""
        k = len(indices)
        total = 3 ** k
        sums = np.zeros(total, dtype=np.int64)
        
        # Precompute weights for this half
        weights = np.array([W[i] for i in indices], dtype=np.int64)
        
        for i in range(total):
            x = i
            s = 0
            for j in range(k):
                digit = x % 3
                x //= 3
                if digit == 0:
                    s -= weights[j]
                elif digit == 2:
                    s += weights[j]
            sums[i] = s
        return sums

    # For 15 elements, 3^15 = 14348901 which is manageable
    # But let's be smarter: use recursive generation with numpy

    def gen_sums_and_coeffs(indices):
        """Generate all sums and their corresponding coefficient patterns for a subset."""
        k = len(indices)
        total = 3 ** k
        sums = np.zeros(total, dtype=np.int64)
        coeffs_arr = np.zeros((total, k), dtype=np.int8)
        
        weights = np.array([W[i] for i in indices], dtype=np.int64)
        
        # Use iterative approach
        for i in range(total):
            x = i
            s = 0
            for j in range(k):
                digit = x % 3
                x //= 3
                if digit == 0:
                    c = -1
                    s -= weights[j]
                elif digit == 1:
                    c = 0
                else:
                    c = 1
                    s += weights[j]
                coeffs_arr[i, j] = c
            sums[i] = s
        return sums, coeffs_arr

    # Check time
    if time.time() > deadline - 10:
        # Fallback to simpler search
        return _fallback_search()

    # Generate for first half (indices 0-14)
    half1_indices = list(range(half))
    half2_indices = list(range(half, n))

    sums1, coeffs1 = gen_sums_and_coeffs(half1_indices)
    sums2, coeffs2 = gen_sums_and_coeffs(half2_indices)

    # Sort sums2 and keep track of original indices
    sort_order2 = np.argsort(sums2)
    sums2_sorted = sums2[sort_order2]
    coeffs2_sorted = coeffs2[sort_order2]

    # For each sum in half1, find the closest sum in half2 such that sum1 + sum2 != 0
    # and minimize |sum1 + sum2|
    
    # Binary search approach
    best_total_abs = float('inf')
    best_c1 = None
    best_c2 = None

    # Vectorized approach: for each sum1, find the two closest values in sums2_sorted
    # that make the total nonzero

    # This is still O(3^15 * log(3^15)) which should be fast enough

    for i in range(len(sums1)):
        s1 = sums1[i]
        # We want to find s2 such that |s1 + s2| is minimized and s1 + s2 != 0
        # Binary search for -s1 in sums2_sorted
        target = -s1
        pos = np.searchsorted(sums2_sorted, target)
        
        candidates = []
        if pos < len(sums2_sorted):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        
        for p in candidates:
            s2 = sums2_sorted[p]
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_total_abs:
                best_total_abs = abs_total
                best_c1 = coeffs1[i].copy()
                best_c2 = coeffs2_sorted[p].copy()
                best_N = total

    if best_c1 is not None:
        # Reconstruct full coefficient vector
        full_coeffs = [0] * n
        for j, idx in enumerate(half1_indices):
            full_coeffs[idx] = int(best_c1[j])
        for j, idx in enumerate(half2_indices):
            full_coeffs[idx] = int(best_c2[j])
        
        # Verify
        N = calc_N(full_coeffs)
        if N != 0:
            best_coeffs = full_coeffs
            best_abs_N = abs(N)
            best_N = N
        else:
            # Shouldn't happen, but fallback
            return _fallback_search()

    # If meet-in-the-middle worked, we have the exact optimum
    return best_coeffs


def _fallback_search():
    """Simple fallback if meet-in-the-middle fails or times out."""
    n = 30

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, n + 1))
    W = [L // (i + 1) for i in range(n)]

    def calc_N(coeffs):
        s = 0
        for i in range(n):
            c = coeffs[i]
            if c != 0:
                s += c * W[i]
        return s

    best_N = None
    best_coeffs = None
    best_abs_N = float('inf')

    # Greedy initialization
    coeffs = [0] * n
    order = sorted(range(n), key=lambda i: -W[i])
    for i in order:
        best_c = 0
        best_abs = abs(calc_N(coeffs))
        for c in [-1, 0, 1]:
            if c == 0:
                continue
            coeffs[i] = c
            val = abs(calc_N(coeffs))
            if val < best_abs:
                best_abs = val
                best_c = c
            coeffs[i] = 0
        coeffs[i] = best_c

    N = calc_N(coeffs)
    if N == 0:
        coeffs[0] = 1
        N = calc_N(coeffs)

    # Local search
    improved = True
    while improved:
        improved = False
        for i in range(n):
            old_val = coeffs[i]
            current_abs = abs(N)
            best_new_val = None
            best_new_N = None
            best_new_abs = current_abs

            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                delta = (new_val - old_val) * W[i]
                new_N = N + delta
                if new_N == 0:
                    continue
                new_abs = abs(new_N)
                if new_abs < best_new_abs:
                    best_new_abs = new_abs
                    best_new_val = new_val
                    best_new_N = new_N

            if best_new_val is not None:
                coeffs[i] = best_new_val
                N = best_new_N
                improved = True

    if abs(N) < best_abs_N:
        best_abs_N = abs(N)
        best_coeffs = coeffs[:]
        best_N = N

    if best_coeffs is None:
        best_coeffs = [1] + [0] * (n - 1)

    return best_coeffs
