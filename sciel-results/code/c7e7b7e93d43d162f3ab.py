import sys
from math import isqrt, gcd
from collections import defaultdict

sys.setrecursionlimit(10000)

def construct():
    # Enumerate small k values and collect M = k * sigma(k)
    # We use a sieve-like approach to compute sigma efficiently
    # Then find the M with the most k values mapping to it
    
    # First, let's use a smaller sieve for sigma up to some limit
    # k must be <= sqrt(M) and M <= 10^18, so k <= 10^9
    # But sigma(k) can be large. Let's think about this differently.
    
    # For k to satisfy k * sigma(k) = M with k <= sqrt(M),
    # we need sigma(k) >= sqrt(M) / k >= sqrt(k) since k <= sqrt(M)
    # Actually k * sigma(k) = M and k <= sqrt(M) means sigma(k) >= sqrt(M) >= k
    # So sigma(k) >= k, which is always true for k >= 1 (sigma(k) >= k + 1 for k > 1, sigma(1)=1)
    
    # Let's enumerate k up to a reasonable bound and find collisions
    # The key insight: for highly composite M, many k values may map to it
    
    # Use numpy for faster sigma computation
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False
    
    # We'll enumerate k up to K_MAX and collect M values
    # K_MAX should be large enough to find good collisions
    # k * sigma(k) <= 10^18, and for typical k, sigma(k) is roughly O(k log log k)
    # So k * k * log log k <= 10^18 roughly means k <= 10^9 / log log k
    # But we want k <= sqrt(M) = sqrt(k*sigma(k)) so sigma(k) >= k
    # This is always satisfied, so the constraint is just k * sigma(k) <= 10^18
    
    # For practical purposes, let's go up to k = 10^6 or so with sieve
    # That gives us M up to about 10^6 * 10^7 = 10^13, well within 10^18
    
    K_MAX = 2 * 10**6
    
    if use_numpy:
        sigma = np.zeros(K_MAX + 1, dtype=np.int64)
        for i in range(1, K_MAX + 1):
            sigma[i::i] += i
    else:
        sigma = [0] * (K_MAX + 1)
        for i in range(1, K_MAX + 1):
            for j in range(i, K_MAX + 1, i):
                sigma[j] += i
    
    # Collect M -> list of k
    m_to_k = defaultdict(list)
    
    for k in range(1, K_MAX + 1):
        s = sigma[k]
        M = k * s
        if M <= 10**18:
            m_to_k[M].append(k)
    
    # Find M with most k values
    best_M = 0
    best_count = 0
    best_ks = []
    
    for M, ks in m_to_k.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_ks = ks
    
    # Verify that each k satisfies k <= isqrt(M)
    # and k * sigma(k) == M
    valid_ks = []
    for k in best_ks:
        if k <= isqrt(best_M):
            # Verify k * sigma(k) == best_M
            # We need to recompute sigma(k) exactly
            # But we already have it from the sieve
            s = sigma[k]
            if k * s == best_M:
                valid_ks.append(k)
    
    # Ensure distinctness and proper types
    valid_ks = list(set(valid_ks))
    valid_ks.sort()
    
    # Make sure M is a proper Python int
    M_val = int(best_M)
    k_list = [int(k) for k in valid_ks]
    
    return {'M': M_val, 'k': k_list}
