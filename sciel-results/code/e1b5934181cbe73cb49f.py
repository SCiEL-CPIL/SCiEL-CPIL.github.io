"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 1000000
    spf = [0] * (N + 1)
    primes = []
    
    # Linear sieve for Smallest Prime Factor
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            
    def phi_calc(n):
        if n == 1:
            return 1
        result = n
        temp = n
        while temp > 1:
            p = spf[temp]
            result -= result // p
            while temp % p == 0:
                temp //= p
        return result

    for n in range(1, N + 1):
        if phi_calc(n) % 120 == 0:
            return n
            
    return 14400

# EVOLVE-BLOCK-END
