# EVOLVE-BLOCK-START
import math

def construct():
    # Precompute smallest prime factor (SPF) for all integers up to 1,000,000
    # using a linear sieve.
    LIMIT = 1000000
    spf = list(range(LIMIT + 1))
    spf[0] = 0
    spf[1] = 1
    primes = []
    
    for i in range(2, LIMIT + 1):
        if spf[i] == i:
            primes.append(i)
        for p in primes:
            if i * p > LIMIT:
                break
            spf[i * p] = p
            if i % p == 0:
                break

    # Iterate n from 1 upwards to find the smallest n such that phi(n) % 120 == 0
    # We compute phi(n) on the fly using the precomputed SPF.
    for n in range(1, LIMIT + 1):
        if n == 1:
            phi_n = 1
        else:
            phi_n = n
            temp = n
            last_p = 0
            while temp > 1:
                p = spf[temp]
                if p != last_p:
                    phi_n -= phi_n // p
                    last_p = p
                temp //= p
        
        if phi_n % 120 == 0:
            return n
            
    # Fallback if not found (should not happen for 1,000,000)
    return 14400
# EVOLVE-BLOCK-END
