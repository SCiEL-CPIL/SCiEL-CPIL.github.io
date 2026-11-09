"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute phi(n) for all n up to 1000000.
    """
    limit = 1000000
    # Initialize phi[i] = i
    phi = list(range(limit + 1))
    
    # Sieve to compute phi
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n with phi[n] % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (should not reach here if a solution exists in range)
    return -1
# EVOLVE-BLOCK-END
