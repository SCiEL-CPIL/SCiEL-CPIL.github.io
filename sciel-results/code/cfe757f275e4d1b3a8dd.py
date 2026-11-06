"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # The optimal strategy for maximizing the number of coprime adjacent divisors
    # within a bound is to use the largest square-free number composed of the
    # smallest primes (the primorial), as this maximizes the diversity of divisors
    # and minimizes the likelihood of adjacent divisors sharing factors compared
    # to numbers with repeated prime factors.
    
    limit = 10**18
    
    # List of small primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    n = 1
    for p in primes:
        if n * p <= limit:
            n *= p
        else:
            break
            
    return n
# EVOLVE-BLOCK-END
