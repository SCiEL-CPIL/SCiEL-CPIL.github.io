"""Rewritten program: meet-in-the-middle exact enumeration for minimal nonzero signed harmonic subsum."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from bisect import bisect_left

def _precompute_weights(n=30):
    """Precompute LCM of 1..n and weights w[i] = LCM // (i+1) for i in 0..n-1."""
    lcm = 1
    for i in range(1, n + 1):
        lcm = lcm * i // gcd(lcm, i)
    weights = [lcm // (i + 1) for i in range(n)]
    return weights

def _enumerate_half(weights, start, end, deadline):
    """
    Enumerate all signed sums for indices [start, end) with coefficients in {-1, 0, 1}.
    Returns a dict mapping integer sum -> list of coefficient tuples (for those indices).
    """
    length = end - start
    sums = {}
    
    # Iterative DFS using a stack
    stack = [(start, 0, [0] * (end - start))]
    while stack:
        if time.time() > deadline:
            break
        idx, current_sum, coeffs = stack.pop()
        if idx == end:
            if current_sum not in sums:
                sums[current_sum] = coeffs[:]
            continue
        w = weights[idx]
        # Push options in reverse order so that 0 is processed first (LIFO)
        coeffs[idx] = -1
        stack.append((idx + 1, current_sum - w, coeffs[:]))
        coeffs[idx] = 1
        stack.append((idx + 1, current_sum + w, coeffs[:]))
        coeffs[idx] = 0
        stack.append((idx + 1, current_sum, coeffs[:]))
    
    return sums

def construct():
    deadline = time.time() + 50  # Safety margin before 90s limit
    
    n = 30
    weights = _precompute_weights(n)
    
    # Split into two halves
    mid = n // 2  # 15
    left_weights = weights[:mid]
    right_weights = weights[mid:]
    
    # Enumerate left half: indices 0..14
    left_sums = _enumerate_half(weights, 0, mid, deadline)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Enumerate right half: indices 15..29
    right_sums = _enumerate_half(weights, mid, n, deadline)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # For each left sum L, find right sum R such that L + R != 0 and |L + R| is minimized
    right_keys = sorted(right_sums.keys())
    
    best_total = None
    best_left_sum = None
    best_right_sum = None
    
    for L, left_coeff in left_sums.items():
        if time.time() > deadline:
            break
        # Binary search for -L in right_keys
        pos = bisect_left(right_keys, -L)
        
        # Check positions around pos
        for p in range(max(0, pos - 3), min(len(right_keys), pos + 4)):
            R = right_keys[p]
            total = L + R
            if total == 0:
                continue
            if best_total is None or abs(total) < abs(best_total):
                best_total = total
                best_left_sum = L
                best_right_sum = R
    
    if best_total is None:
        return [0] * 28 + [1, -1]
    
    # Reconstruct coefficients
    left_coeff = left_sums[best_left_sum]
    right_coeff = right_sums[best_right_sum]
    coeffs = left_coeff + right_coeff
    
    # Verify
    total = sum(c * w for c, w in zip(coeffs, weights))
    if total == 0:
        return [0] * 28 + [1, -1]
    
    return coeffs
# EVOLVE-BLOCK-END
