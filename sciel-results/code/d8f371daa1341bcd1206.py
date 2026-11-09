import time
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Sieve of Eratosthenes to get primes up to MAX_N
    is_prime = [True] * (MAX_N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if is_prime[i]:
            for j in range(i * i, MAX_N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, MAX_N + 1) if is_prime[i]]
    
    # Compute powerful_part for each n
    # powerful_part[n] = product of p^e for e >= 2 in factorization of n
    pp = [1] * (MAX_N + 2)  # index up to MAX_N + 1 for n+1
    
    for p in primes:
        p2 = p * p
        if p2 > MAX_N + 1:
            break
        p3 = p2 * p
        # For each multiple of p^2, determine the exact power of p
        for n in range(p2, MAX_N + 2, p2):
            # n is divisible by p^2
            if n % p3 == 0:
                # e >= 3, compute fully
                temp = n
                e = 0
                while temp % p == 0:
                    temp //= p
                    e += 1
                pp[n] *= (p ** e)
            else:
                # e == 2 exactly
                pp[n] *= p2
    
    # Now scan for best n in [MIN_N, MAX_N]
    best_n = MIN_N
    best_score = pp[MIN_N] * pp[MIN_N + 1]
    
    # Fixed range to include MAX_N
    for n in range(MIN_N, MAX_N + 1):
        score = pp[n] * pp[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
        if time.time() > DEADLINE:
            break
    
    return best_n
