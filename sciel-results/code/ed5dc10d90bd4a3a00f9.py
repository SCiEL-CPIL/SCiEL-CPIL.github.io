import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # need n and n+1, so up to 1000001
    
    # powerful_part[i] will hold the powerful part of i
    # powerful_part[i] = product of p^e for each prime p with exponent e >= 2 in i's factorization
    powerful_part = [1] * (MAX_VAL + 1)
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each prime p, for each power p^e with e >= 2,
    # multiply p^e into powerful_part[m] for all m divisible by p^e
    # We do this by iterating over primes and their powers
    
    # First, collect all primes
    primes = []
    for i in range(2, MAX_VAL + 1):
        if spf[i] == i:
            primes.append(i)
    
    # For each prime, compute all powers p^e with e >= 2 that are <= MAX_VAL
    # and multiply them into powerful_part for multiples
    for p in primes:
        # Start with p^2
        power = p * p
        while power <= MAX_VAL:
            # For all multiples of power, multiply by power
            for m in range(power, MAX_VAL + 1, power):
                powerful_part[m] *= power
            power *= p
    
    # Find the maximum product of consecutive powerful parts
    best_n = 10000
    best_product = powerful_part[10000] * powerful_part[10001]
    
    for n in range(10000, N):
        prod = powerful_part[n] * powerful_part[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
        
        # Check deadline
        if time.time() > deadline - 2.0:
            break
    
    return best_n
