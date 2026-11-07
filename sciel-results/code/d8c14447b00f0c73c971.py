import math
from collections import defaultdict

def construct():
    # We'll search for M <= 10^18 with many k <= sqrt(M) such that k*sigma(k) = M
    # Strategy: enumerate k up to a limit, compute k*sigma(k), count multiplicities
    # Use a sieve to compute sigma(k) efficiently
    
    # Set a reasonable limit for k enumeration
    # k <= sqrt(M) and M <= 10^18, so k <= 10^9
    # But we can't enumerate to 10^9, so we'll enumerate to a practical limit
    # and hope to find good M values
    
    # Let's try enumerating k up to 2 * 10^6 first
    N = 2000000
    
    # Sieve for sigma function
    # sigma(n) = sum of divisors
    # We can compute it using a modified sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Now compute k * sigma(k) for each k and count
    m_counts = defaultdict(list)
    for k in range(1, N + 1):
        val = k * sigma[k]
        if val <= 10**18:
            m_counts[val].append(k)
    
    # Find M with most solutions
    best_M = 1
    best_k_list = [1]
    for m, ks in m_counts.items():
        # Filter k <= sqrt(m)
        sqrt_m = math.isqrt(m)
        valid_ks = [k for k in ks if k <= sqrt_m]
        if len(valid_ks) > len(best_k_list):
            best_M = m
            best_k_list = valid_ks
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    return {'M': best_M, 'k': best_k_list}
