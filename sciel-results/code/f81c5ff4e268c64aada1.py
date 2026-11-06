import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 1000000
    LOW = 10000
    
    # Segmented sieve to compute powerful part for every number in [LOW, N]
    # powerful_part(n) = product of p^e where e >= 2 in prime factorization of n
    
    # First, get all primes up to sqrt(N) = 1000
    limit = int(math.isqrt(N)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    
    # Array to hold powerful part for each index in [LOW, N]
    # We use 1-indexed offset: pw[i] corresponds to number LOW + i
    size = N - LOW + 1
    pw = [1] * size
    
    # For each prime p, for each power p^e (e>=2) that divides numbers in range,
    # multiply pw[idx] by p^e for all multiples of p^e in the range.
    # But we need the FULL prime power, so for each number, we need to find the
    # highest power of each prime that divides it.
    #
    # Strategy: for each prime p, iterate through powers p^2, p^3, p^4, ...
    # For p^e (e>=2), for every multiple m of p^e in [LOW, N], 
    # we need to check if p^(e+1) does NOT divide m (i.e., the exact exponent is e).
    # Actually, a simpler approach: for each prime p, for each multiple of p^2 in range,
    # divide out all factors of p to get the exact power, then multiply pw by that power.
    
    # Better approach: for each prime p, for each multiple k of p in [LOW, N],
    # count the exact exponent of p in k, and if exponent >= 2, multiply pw by p^exponent.
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # For each multiple of p in [LOW, N], compute exact power of p
        # Start from the first multiple of p >= LOW
        first_mult = ((LOW + p - 1) // p) * p
        for m in range(first_mult, N + 1, p):
            # Count exact exponent of p in m
            count = 0
            temp = m
            while temp % p == 0:
                count += 1
                temp //= p
            if count >= 2:
                pw[m - LOW] *= p ** count
    
    # Now find the n in [LOW, N-1] that maximizes pw[n-LOW] * pw[n+1-LOW]
    best_n = LOW
    best_val = 0
    for i in range(size - 1):
        val = pw[i] * pw[i + 1]
        if val > best_val:
            best_val = val
            best_n = LOW + i
    
    # Check deadline
    if time.time() > deadline - 2.0:
        return best_n
    
    return best_n
