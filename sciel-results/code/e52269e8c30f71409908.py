import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Strategy: Direct enumeration of k up to a limit N, computing sigma(k) via sieve.
    # Find the M with the most preimages k.
    
    # We can sieve up to N. Let's try N = 5,000,000.
    # sigma(n) can be computed in O(N log N) using a modified sieve.
    
    N = 5000000
    
    # Check if we have time. Sieve of 5M is very fast.
    if time.time() > deadline - 10:
        # Fallback to hardcoded if extremely tight (shouldn't happen)
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Sieve for sigma
    # sigma[n] = sum of divisors of n
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    # Now map M = k * sigma(k) to list of k
    m_to_k = defaultdict(list)
    
    # We only care about k such that k * sigma(k) <= 10^18
    # Also k must be <= sqrt(M). 
    # If M = k*sigma(k), then k <= sqrt(M) implies k^2 <= M = k*sigma(k) => k <= sigma(k).
    # This is always true for k >= 2? 
    # sigma(k) >= k+1 for k>1. So k < sigma(k) is always true.
    # So the condition k <= sqrt(M) is automatically satisfied for any k > 1.
    # Wait, sqrt(M) = sqrt(k*sigma(k)).
    # k <= sqrt(k*sigma(k))  <=> k^2 <= k*sigma(k) <=> k <= sigma(k).
    # Since sigma(k) >= 1 + k, this holds for all k >= 1.
    # So any k found in the sieve is a valid candidate for k_i.
    
    max_solutions = 0
    best_M = 0
    best_ks = []
    
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M > 10**18:
            # Since k and sigma(k) are increasing, we could break, but sigma(k) isn't strictly increasing with k.
            # However, for large k, M will exceed 10^18.
            # Let's just skip.
            continue
            
        m_to_k[M].append(k)
        
        # Check if this is the new best
        if len(m_to_k[M]) > max_solutions:
            max_solutions = len(m_to_k[M])
            best_M = M
            best_ks = list(m_to_k[M])
            
        # Time check
        if time.time() > deadline - 5:
            break
            
    # If we found a better solution
    if max_solutions > 3:
        return {'M': best_M, 'k': best_ks}
    else:
        # Return the known best from seed
        return {'M': 333312, 'k': [336, 372, 434]}
