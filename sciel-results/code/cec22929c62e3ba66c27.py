"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute Euler's totient function for all numbers up to 1,000,000.
    """
    N = 1000000
    # Initialize phi array
    phi = list(range(N + 1))
    
    # Sieve for totient
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    return -1 # Should not reach here
# EVOLVE-BLOCK-END
