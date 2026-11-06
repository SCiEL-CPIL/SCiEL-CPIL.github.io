"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The score tau_perp(n) depends on the structure of the divisors.
    # For n = p1 * p2 * ... * pk (distinct primes), the divisors correspond to subsets of {1..k}.
    # Two divisors are coprime if their corresponding subsets are disjoint.
    # The maximum number of coprime adjacent pairs is maximized by maximizing k,
    # the number of distinct prime factors, subject to n <= 10^18.
    #
    # The product of the first 15 primes is approx 6.15 * 10^17.
    # The product of the first 16 primes is approx 9.7 * 10^18, which exceeds 10^18.
    # Thus, the maximum number of distinct prime factors is 15.
    # Any set of 15 distinct primes yields the same graph structure and thus the same maximum score.
    # We use the first 15 primes for simplicity.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
        
    # Verify n <= 10^18
    if n > 10**18:
        # Fallback, though not expected
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
        
    return n
# EVOLVE-BLOCK-END
