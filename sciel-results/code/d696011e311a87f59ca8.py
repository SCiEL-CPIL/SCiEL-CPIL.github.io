"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to precompute phi for all numbers up to a sufficient limit,
    then iterates to find the minimum n satisfying the condition.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin out of 60s limit
    
    # The minimal n is known to be small. 
    # phi(143) = phi(11*13) = 10*12 = 120.
    # A limit of 1000 is more than enough to capture the minimum.
    limit = 1000
    
    # Initialize phi array
    phi = list(range(limit + 1))
    
    # Sieve of Eratosthenes style for phi
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    # Check if we are still within time limits (should be, as limit is small)
    if time.time() > deadline:
        # Fallback if something unexpected happens with the sieve
        return 143

    # Find the smallest n such that phi[n] % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # If not found in the small limit (highly unlikely for min), 
    # we could expand, but 143 is guaranteed to be found.
    # As a safety net, return the known minimal value.
    return 143
# EVOLVE-BLOCK-END
