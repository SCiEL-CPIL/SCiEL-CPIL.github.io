"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def _sieve_sigma(limit):
    """Compute sigma(n) for all n <= limit using a sieve."""
    sigma = [0] * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # We want to find M with many k such that k * sigma(k) = M.
    # Strategy: enumerate k up to some limit, compute M_k = k * sigma(k),
    # group by M_k, and find the M with the most k's.
    #
    # Key insight: for small k, sigma(k) is roughly O(k log log k), so
    # M_k = k * sigma(k) is roughly O(k^2 log log k).
    # To get M <= 10^18, we need k <= sqrt(10^18 / log log k) ~ 10^9 / sqrt(log log k)
    # But we can't enumerate up to 10^9.
    #
    # However, the finding says "Direct sieve enumeration up to 5M finds collisions
    # in the dense low-range where sigma(k) is small and k values are abundant,
    # likely uncovering M with 4+ solutions".
    #
    # So let's enumerate k up to a reasonable limit and find M with many solutions.
    # We need k_i <= sqrt(M), and M = k_i * sigma(k_i).
    # If k is small, M is small, and sqrt(M) might be smaller than some other k' that
    # also gives the same M. We need to check k' <= sqrt(M).
    
    # Let's try different limits and see what we find.
    # For each M, we collect all k such that k*sigma(k)=M and k <= sqrt(M).
    
    best_M = None
    best_k_list = []
    
    # Try progressively larger limits
    limits = [100000, 500000, 2000000, 5000000, 10000000]
    
    for limit in limits:
        if time.time() > deadline - 5.0:
            break
        
        sigma = _sieve_sigma(limit)
        
        # Group k by M = k * sigma(k)
        m_to_ks = defaultdict(list)
        for k in range(1, limit + 1):
            M_val = k * sigma[k]
            if M_val <= 10**18:
                m_to_ks[M_val].append(k)
        
        # Find the M with the most valid k's (where k <= sqrt(M))
        for M_val, ks in m_to_ks.items():
            sqrt_M = math.isqrt(M_val)
            valid_ks = [k for k in ks if k <= sqrt_M]
            if len(valid_ks) > len(best_k_list):
                best_k_list = valid_ks
                best_M = M_val
        
        # Early exit if we found a good one
        if len(best_k_list) >= 10:
            break
    
    if best_M is None:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    return {'M': best_M, 'k': best_k_list}
# EVOLVE-BLOCK-END
