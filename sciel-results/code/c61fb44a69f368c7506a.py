"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1  # We need powerful_part for n and n+1, so up to 1000001
    
    # Sieve of Eratosthenes to find primes up to sqrt(LIMIT)
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = [True] * (sqrt_limit + 1)
    is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, sqrt_limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Initialize powerful part array
    # pp[x] will store the powerful part of x
    pp = [1] * (LIMIT + 1)
    
    # For each prime, process its powers
    for p in primes:
        p2 = p * p
        if p2 > LIMIT:
            break
        # For each power p^e where e >= 2 and p^e <= LIMIT
        pe = p2
        while pe <= LIMIT:
            # For each multiple of pe, we need to determine the exact exponent of p in that number
            # If pe divides the number but p^(e+1) does not, then the exponent is exactly e
            # and we multiply pp by p^e
            pe_next = pe * p
            for multiple in range(pe, LIMIT + 1, pe):
                # Check if p^(e+1) divides multiple
                if multiple % pe_next != 0:
                    pp[multiple] *= pe
            pe = pe_next
    
    # Scan for the best n in [N_MIN, N_MAX] maximizing pp[n] * pp[n+1]
    best_n = N_MIN
    best_product = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX + 1):
        prod = pp[n] * pp[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
