"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    from math import isqrt
    # Find all primes in 2..40
    primes = []
    for a in range(2, P['n'] + 1):
        is_prime = True
        for d in range(2, isqrt(a) + 1):
            if a % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(a)
    # Verify all subset products are distinct (safety check)
    # For primes, this is guaranteed by unique factorization
    return primes
# EVOLVE-BLOCK-END
