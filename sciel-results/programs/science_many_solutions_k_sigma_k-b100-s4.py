"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _isqrt(x):
    return math.isqrt(x)

def _sigma_sieve(n):
    """Compute sigma(k) for k=1..n using a sieve."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def construct():
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Strategy: For each k, compute M_k = k * sigma(k).
    # We want M_k values that repeat many times with k <= sqrt(M_k).
    # 
    # Key observation: if k * sigma(k) = M, then k <= sqrt(M) means k <= sqrt(k*sigma(k)),
    # i.e., k^2 <= k*sigma(k), i.e., k <= sigma(k). Since sigma(k) >= k+1 for k>1,
    # this is always true.
    
    # Start with a known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = len(best_k)
    
    # Try to find better solutions by scanning larger ranges
    # First, try a moderate range to find collisions
    for N in [10**5, 2*10**5, 5*10**5, 10**6, 2*10**6, 5*10**6]:
        if time.time() > deadline - 20:
            break
        
        sigma = _sigma_sieve(N)
        
        # Group by M value
        groups = defaultdict(list)
        for k in range(1, N + 1):
            M = k * sigma[k]
            if M <= 10**18 and k <= _isqrt(M):
                groups[M].append(k)
        
        # Find the best M with most solutions
        for M, ks in groups.items():
            count = len(ks)
            if count > best_count:
                best_M = M
                best_k = ks
                best_count = count
        
        # Early exit if we find a very good solution
        if best_count >= 20:
            break
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    # Verify the solution
    for k in best_k:
        if k > _isqrt(best_M):
            return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
