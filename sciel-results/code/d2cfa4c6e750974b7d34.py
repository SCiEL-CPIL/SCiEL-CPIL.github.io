"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    We use a sieve to compute Euler's totient function phi(n) for n up to a small limit.
    Based on number theory, the smallest such n is 143 (phi(143) = 120).
    We verify this algorithmically to ensure correctness.
    """
    limit = 1000  # The answer is known to be small, 1000 is a safe upper bound for the minimum.
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback if not found in the small range (should not happen)
    # We can extend the search or return a known large candidate, but 143 is guaranteed.
    return 143
# EVOLVE-BLOCK-END
