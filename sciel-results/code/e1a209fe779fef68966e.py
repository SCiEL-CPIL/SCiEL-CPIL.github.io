"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import Counter

def construct():
    N = 10_000_000
    sigma = [1] * (N + 1)
    # Sieve for sigma(k)
    for i in range(2, N + 1):
        if sigma[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                power = i
                s = 1
                while j % power == 0:
                    s += power
                    power *= i
                sigma[j] *= s
    
    # Count frequencies of f(k) = k * sigma(k)
    freq = Counter()
    for k in range(1, N + 1):
        f = k * sigma[k]
        if f <= 10**18:
            freq[f] += 1
    
    if not freq:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Find M with max frequency
    best_M = max(freq, key=lambda m: freq[m])
    
    # Collect all k that map to best_M and satisfy k <= sqrt(M)
    k_list = []
    sqrt_M = math.isqrt(best_M)
    for k in range(1, min(N, sqrt_M) + 1):
        if k * sigma[k] == best_M:
            k_list.append(k)
    
    k_list = sorted(set(k_list))
    
    if not k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': k_list}
# EVOLVE-BLOCK-END
