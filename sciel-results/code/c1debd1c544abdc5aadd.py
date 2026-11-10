import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    MAX_N = 1000000
    LIMIT = MAX_N + 1
    
    # 1. Sieve of Smallest Primes (SPF)
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # Helper to compute powerful part using SPF
    def powerful_part(x):
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            term = 1
            while x % p == 0:
                x //= p
                e += 1
                term *= p
            if e >= 2:
                pp *= term
        return pp

    best_n = 10000
    best_score = 0
    
    # 2. Iterate through all valid n
    # We can optimize by precomputing powerful parts for all numbers in range
    # but computing on the fly is also fast enough. 
    # Let's precompute for all numbers up to LIMIT to avoid repeated work if we were to do more, 
    # but here we just need n and n+1.
    # Actually, precomputing an array of powerful parts for all i in [10000, LIMIT] is cleaner and faster.
    
    pp_arr = [0] * (LIMIT + 1)
    for i in range(10000, LIMIT + 1):
        pp_arr[i] = powerful_part(i)
        
    # Check time
    if time.time() > deadline:
        return best_n

    for n in range(10000, MAX_N + 1):
        score = pp_arr[n] * pp_arr[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline:
            break

    return best_n
