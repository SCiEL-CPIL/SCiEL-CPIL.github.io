"""Find M <= 1e18 with many distinct k <= sqrt(M) such that k*sigma(k)=M."""
import math
import time
from collections import defaultdict

def isqrt(n):
    return math.isqrt(n)

def sieve_smallest_prime(limit):
    """Return spf array: spf[i] = smallest prime factor of i."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, isqrt(limit) + 1):
        if spf[i] == i:
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf

def sigma_fast(n, spf):
    """Compute sigma(n) using smallest-prime-factor table. n must be < len(spf)."""
    if n <= 1:
        return n
    result = 1
    m = n
    while m > 1:
        p = spf[m]
        power = 1
        sigma_part = 1
        while m % p == 0:
            m //= p
            power *= p
            sigma_part += power
        result *= sigma_part
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (limit 120s)
    
    # Strategy: compute M = k*sigma(k) for many k, group by M,
    # find M with most distinct k values.
    
    # Precompute SPF up to a limit that fits in time.
    # 10^7 is a good balance. Sieve takes ~1s, loop takes ~5s.
    SPF_LIMIT = 10_000_000
    
    spf = sieve_smallest_prime(SPF_LIMIT)
    
    # Build M -> list of k
    # Using list is faster than set for appending, and we know k is unique in the loop.
    m_to_k = defaultdict(list)
    
    best_M = None
    best_count = 0
    
    # Loop k from 1 to SPF_LIMIT
    for k in range(1, SPF_LIMIT + 1):
        # Check time every 100k iterations to reduce overhead
        if k % 100000 == 0 and time.time() > deadline - 5:
            break
            
        s = sigma_fast(k, spf)
        M = k * s
        
        # M <= 10^18 check
        if M > 10**18:
            continue
            
        # k <= sqrt(M) is always true for k>=1 because sigma(k) >= k+1 for k>1, sigma(1)=1.
        # So M >= k^2 for all k.
        
        m_to_k[M].append(k)
        
        # Update best on the fly to avoid iterating over the huge dict later
        count = len(m_to_k[M])
        if count > best_count:
            best_count = count
            best_M = M
            
    if best_M is None:
        # Fallback if something went wrong or no solutions found (unlikely)
        best_M = 1
        best_k_list = [1]
    else:
        best_k_list = m_to_k[best_M]
        
    # Ensure types are Python int
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    # Deduplicate and sort (though they should be unique and appended in order)
    # Sorting ensures consistency
    best_k_list = sorted(set(best_k_list))
    
    # Truncate to 1000
    best_k_list = best_k_list[:1000]
    
    return {'M': best_M, 'k': best_k_list}
