"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1
    
    # Sieve of Eratosthenes to find primes up to sqrt(LIMIT)
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = [True] * (sqrt_limit + 1)
    is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Initialize powerful part array
    pp = [1] * (LIMIT + 1)
    
    # For each prime, process its powers
    for p in primes:
        p_power = p * p
        while p_power <= LIMIT:
            p_next_power = p_power * p
            # Multiples of p_power that are NOT multiples of p_next_power
            # j * p_power, where j % p != 0
            # j ranges from 1 to LIMIT // p_power
            # Step p, skipping every p-th
            # Actually, iterating all and checking is fine for small N
            # But we can iterate j = 1, 2, ..., step 1
            # Optimization: iterate j, if j%p!=0 update
            
            limit_j = LIMIT // p_power
            for j in range(1, limit_j + 1):
                if j % p != 0:
                    x = j * p_power
                    pp[x] *= p_power
            
            p_power = p_next_power
            
    # Find n in [N_MIN, N_MAX] that maximizes pp[n] * pp[n+1]
    best_n = N_MIN
    max_product = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        prod = pp[n] * pp[n + 1]
        if prod > max_product:
            max_product = prod
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
