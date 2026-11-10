# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve up to MAX_N + 1
    limit = MAX_N + 1
    spf = list(range(limit + 1))
    
    # Sieve of Eratosthenes variant for SPF
    # Only need to iterate up to sqrt(limit)
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number in [1, MAX_N+1]
    # Powerful part = product of p^e for each prime factor p with exponent e >= 2
    pp = [1] * (limit + 1)
    
    # Optimization: Use spf[x] == p check instead of x % p == 0
    for n in range(2, limit + 1):
        x = n
        val = 1
        while x > 1:
            p = spf[x]
            e = 1
            x //= p
            # Continue dividing by p as long as spf[x] is p
            while x > 1 and spf[x] == p:
                e += 1
                x //= p
            if e >= 2:
                val *= (p ** e)
        pp[n] = val
        
        if time.time() > deadline:
            # If we time out during precomputation, we cannot guarantee correctness of the search
            # But we must return something. 
            # However, precomputation should be fast.
            # If it times out, we might have incomplete pp array.
            # To be safe, if we haven't finished precomputation, we should probably return a known good value or break.
            # But the loop is 1..1e6. It should finish quickly.
            pass

    # Find n in [MIN_N, MAX_N] that maximizes pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = pp[MIN_N] * pp[MIN_N + 1]
    
    for n in range(MIN_N, MAX_N):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
