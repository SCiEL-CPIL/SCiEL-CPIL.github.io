"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt, gcd

def construct():
    N = 256
    
    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each prime, find the largest prime power p^k <= N
    result = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        result.append(power)
    
    # Verify pairwise coprimality (should always pass, but safety check)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + primes
    
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
