import time

def construct():
    deadline = time.time() + 80.0
    
    N = 1000000
    lo = 10000
    hi = N
    hi1 = hi + 1
    
    # Sieve for primes up to sqrt(hi1)
    sqrt_hi1 = int(hi1 ** 0.5) + 1
    is_prime = bytearray(b'\x01') * (sqrt_hi1 + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(sqrt_hi1 ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, sqrt_hi1 + 1, i):
                is_prime[j] = 0
    primes = [i for i in range(2, sqrt_hi1 + 1) if is_prime[i]]
    
    # Compute largest prime factor for all numbers up to hi1
    # Process primes in increasing order; for each prime p, mark all multiples.
    # The last (largest) prime to divide m is lpf[m].
    lpf = [1] * (hi1 + 1)
    for p in primes:
        for m in range(p, hi1 + 1, p):
            lpf[m] = p
    
    # Any x > 1 not marked by any prime <= sqrt(x) is itself prime
    for x in range(2, hi1 + 1):
        if lpf[x] == 1:
            lpf[x] = x
    
    # Find n in [lo, hi] minimizing max(lpf[n], lpf[n+1])
    best_n = lo
    best_val = max(lpf[lo], lpf[lo + 1])
    
    for n in range(lo + 1, hi + 1):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
