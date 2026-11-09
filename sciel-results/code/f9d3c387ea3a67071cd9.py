"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve-based approach to pre-compute the smallest prime factor (SPF)
    for all numbers up to 1,000,000, then efficiently computes phi(n) for
    increasing n until the condition is met.
    """
    limit = 1000000
    
    # Sieve of Eratosthenes variant to compute smallest prime factor for each number
    spf = [0] * (limit + 1)
    spf[1] = 1
    for i in range(2, limit + 1):
        if spf[i] == 0:
            # i is prime
            for j in range(i, limit + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    
    # Helper to compute phi(n) using SPF
    def compute_phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        while temp > 1:
            p = spf[temp]
            # Remove all factors of p
            while temp % p == 0:
                temp //= p
            result -= result // p
        return result
    
    # Search for the smallest n
    # We start from 1, but we know phi(1)=1, phi(2)=1, etc.
    # We can optimize the search start, but a simple loop is fast enough with SPF.
    # phi(n) is even for n > 2.
    # 120 = 2^3 * 3 * 5.
    
    for n in range(1, limit + 1):
        if compute_phi(n) % 120 == 0:
            return n
            
    # Fallback, should not happen
    return -1
# EVOLVE-BLOCK-END
