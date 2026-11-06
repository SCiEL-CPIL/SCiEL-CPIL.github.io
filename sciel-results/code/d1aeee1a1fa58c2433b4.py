"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Precompute smallest prime factor (SPF) up to 1,000,000 using a linear sieve
    N = 1000000
    spf = [0] * (N + 1)
    primes = []
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            
    # Iterate n from 1 upwards to find the smallest n with phi(n) % 120 == 0
    for n in range(1, N + 1):
        if n == 1:
            phi_n = 1
        else:
            phi_n = n
            temp = n
            while temp > 1:
                p = spf[temp]
                phi_n -= phi_n // p
                while temp % p == 0:
                    temp //= p
        if phi_n % 120 == 0:
            return n
            
    return -1 # Should not reach here

# EVOLVE-BLOCK-END
