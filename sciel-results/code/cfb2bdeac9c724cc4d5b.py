"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The product of the first 15 primes is 614889782588491410, which is <= 10^18.
    # The product of the first 16 primes is > 10^18.
    # Using the maximum number of distinct prime factors (primorial) maximizes
    # the number of divisors and typically the number of coprime adjacent pairs.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
# EVOLVE-BLOCK-END
