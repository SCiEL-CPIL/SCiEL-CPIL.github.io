# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve up to MAX_N + 1
    limit = MAX_N + 1
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number in [1, MAX_N+1]
    pp = [1] * (limit + 1)
    
    for n in range(2, limit + 1):
        x = n
        val = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                val *= p ** e
        pp[n] = val
    
    # Find the n in [MIN_N, MAX_N] that maximizes pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = pp[MIN_N] * pp[MIN_N + 1]
    
    for n in range(MIN_N, MAX_N):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
    
    # Check deadline
    if time.time() > deadline:
        return best_n
    
    return best_n
# EVOLVE-BLOCK-END
