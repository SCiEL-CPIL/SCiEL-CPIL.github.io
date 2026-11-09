"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1  # we need n+1 up to 1000001
    
    # Find primes up to sqrt(LIMIT) = 1000
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = [True] * (sqrt_limit + 1)
    is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    
    # Initialize powerful part array: pp[x] = powerful part of x
    # Powerful part = product of p^e for all prime powers p^e || x where e >= 2
    pp = [1] * (LIMIT + 1)
    
    # For each prime p, for each power p^k (k>=2), multiply pp[multiple] by p^k
    # if p^k divides the multiple
    for p in primes:
        p2 = p * p
        if p2 > LIMIT:
            break
        pk = p2
        while pk <= LIMIT:
            # For all multiples of pk, pk contributes to their powerful part
            for m in range(pk, LIMIT + 1, pk):
                pp[m] *= pk
            pk *= p
            if pk > LIMIT:
                break
    
    # Now find n in [N_MIN, N_MAX] that maximizes pp[n] * pp[n+1]
    best_n = N_MIN
    best_val = 0
    for n in range(N_MIN, N_MAX + 1):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    # Check deadline
    if time.time() > DEADLINE:
        return best_n
    
    return best_n
# EVOLVE-BLOCK-END
