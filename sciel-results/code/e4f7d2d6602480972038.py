import time
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    TIME_LIMIT = 300
    primes = _primes(N)
    num_primes = len(primes)
    
    counts = [[0] * p for p in primes]
    non_zero = [0] * num_primes
    missing = [-1] * num_primes
    
    S = set()
    
    candidates = list(range(N))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        K_new = len(S) + 1
        
        valid = True
        for i, p in enumerate(primes):
            if p > K_new:
                break
            if non_zero[i] == p - 1:
                if missing[i] == x % p:
                    valid = False
                    break
        
        if valid:
            S.add(x)
            for i, p in enumerate(primes):
                r = x % p
                if counts[i][r] == 0:
                    non_zero[i] += 1
                    if non_zero[i] == p - 1:
                        for r2 in range(p):
                            if counts[i][r2] == 0:
                                missing[i] = r2
                                break
                counts[i][r] += 1
                
    return list(S)
