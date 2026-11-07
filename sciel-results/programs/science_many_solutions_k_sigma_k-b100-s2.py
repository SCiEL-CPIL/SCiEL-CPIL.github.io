import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 110  # Safety margin
    
    # Strategy: Use a sieve to compute sigma(k) for k up to N.
    # Then find M values that are k*sigma(k) for many k.
    
    # Estimate N based on time. 
    # Sieve complexity is O(N log N). 
    # Let's try N = 2,000,000 first. If it's too slow, we'll catch it or just set a limit.
    # Actually, a simple sieve for sigma is quite fast in Python if optimized.
    # Let's start with N = 1,000,000 and see. We can increase if time permits.
    
    # To be safe and fast, we'll use a dynamic approach or a fixed large N.
    # A sieve up to 2 million is typically fast (< 2-3 seconds in PyPy, maybe 5-10s in CPython).
    # Let's aim for N = 2,000,000.
    
    N = 2_000_000
    
    # Check if we have enough time for this N.
    # If not, reduce N.
    # For now, assume 2M is okay. If it times out, the fallback will handle it.
    
    sigma = [0] * (N + 1)
    
    # Sieve for sigma
    # For each d, add d to sigma[j] for all j multiple of d
    for d in range(1, N + 1):
        for j in range(d, N + 1, d):
            sigma[j] += d
            
    # Check time
    if time.time() > deadline - 20:
        # If we are running out of time, just return best found so far (which is empty)
        # But we haven't found any yet. Let's use a smaller N or fallback.
        pass

    # Find collisions
    M_to_ks = defaultdict(list)
    best_M = 333312
    best_sols = [336, 372, 434] # Fallback from seed
    
    for k in range(1, N + 1):
        if time.time() > deadline - 10:
            break
        M_val = k * sigma[k]
        if M_val <= 10**18:
            M_to_ks[M_val].append(k)
            # Update best if this M has more solutions
            if len(M_to_ks[M_val]) > len(best_sols):
                best_sols = M_to_ks[M_val]
                best_M = M_val
                
    # Final verification
    # Ensure distinct
    best_sols = list(set(best_sols))
    limit = math.isqrt(best_M)
    valid_sols = []
    for k in best_sols:
        if 1 <= k <= limit:
            # Re-verify sigma just in case, though sieve is exact
            # We can reuse the sieve value if k <= N
            if k <= N:
                s = sigma[k]
            else:
                # This shouldn't happen since k <= N
                s = _sigma_slow(k) 
            if k * s == best_M:
                valid_sols.append(k)
                
    if not valid_sols:
        best_M = 333312
        valid_sols = [336, 372, 434]
        
    return {'M': int(best_M), 'k': [int(x) for x in valid_sols]}

def _sigma_slow(n):
    """Fallback sigma calculation for large n if needed."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p ** (exp + 1) - 1) // (p - 1)
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result
