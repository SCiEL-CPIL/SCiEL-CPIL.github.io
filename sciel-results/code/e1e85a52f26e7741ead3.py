"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import lcm
import bisect
import itertools

def construct():
    # Compute LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    
    # Precompute weights: w[i] = L // i for i=1..30
    weights = [L // i for i in range(1, 31)]
    
    # Split into two halves: indices 0..14 and 15..29
    half1_weights = weights[0:15]
    half2_weights = weights[15:30]
    
    def enumerate_half(weights_list):
        n = len(weights_list)
        seen = {}
        for coeffs in itertools.product([-1, 0, 1], repeat=n):
            s = sum(c * w for c, w in zip(coeffs, weights_list))
            if s not in seen:
                seen[s] = coeffs
        return seen
    
    half1_sums = enumerate_half(half1_weights)
    half2_sums = enumerate_half(half2_weights)
    
    half2_keys = sorted(half2_sums.keys())
    
    best_abs = None
    best_coeffs = None
    
    for s1, coeffs1 in half1_sums.items():
        target = -s1
        pos = bisect.bisect_left(half2_keys, target)
        
        for p in range(max(0, pos-3), min(len(half2_keys), pos+4)):
            s2 = half2_keys[p]
            total = s1 + s2
            if total == 0:
                continue
            if best_abs is None or abs(total) < best_abs:
                best_abs = abs(total)
                coeffs2 = half2_sums[s2]
                best_coeffs = list(coeffs1) + list(coeffs2)
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
