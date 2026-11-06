import math
import time
import numpy as np

def _powerful_part_direct(n):
    """Compute powerful part of n by direct trial division."""
    if n < 4:
        return 1
    result = 1
    temp = n
    # Check factor 2
    e = 0
    while temp % 2 == 0:
        temp //= 2
        e += 1
    if e >= 2:
        result *= 2 ** e
    # Check odd factors
    d = 3
    while d * d <= temp:
        if temp % d == 0:
            e = 0
            while temp % d == 0:
                temp //= d
                e += 1
            if e >= 2:
                result *= d ** e
        d += 2
    # temp is now 1 or a prime with exponent 1
    return result

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1
    
    # Sieve primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Initialize powerful part array P with 1s
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Compute powerful parts using vectorized operations
    for p in primes:
        if time.time() - start_time > deadline - 10:
            break
        p = int(p)
        p2 = p * p
        if p2 > MAX_VAL:
            break
        
        power = p2
        k = 2
        while power <= MAX_VAL:
            if time.time() - start_time > deadline - 10:
                break
            
            next_power = power * p
            
            # Find indices n = m * power where m is not divisible by p
            max_m = MAX_VAL // power
            if max_m >= 1:
                # Vectorized mask approach
                ms = np.arange(1, max_m + 1)
                mask = (ms % p) != 0
                valid_m = ms[mask]
                indices = valid_m * power
                P[indices] *= power
            
            power = next_power
            k += 1
    
    if time.time() - start_time > deadline - 5:
        return 10000
    
    # Compute products
    products = P[10000:1000001] * P[10001:1000002]
    
    best_n = 10000
    best_val = 0
    
    # Check top candidates from sieve
    top_k = 200
    if len(products) < top_k:
        top_k = len(products)
    
    top_indices = np.argpartition(-products, top_k)[:top_k]
    
    for idx in top_indices:
        if time.time() - start_time > deadline - 2:
            break
        n = 10000 + idx
        pn_direct = _powerful_part_direct(n)
        pn1_direct = _powerful_part_direct(n + 1)
        val = pn_direct * pn1_direct
        if val > best_val:
            best_val = val
            best_n = n
            
    # Additional check: Squares in range
    # If n is a square, P(n) = n.
    # We want P(n+1) to be large.
    # Check squares near the upper bound.
    for sq in range(1000, 99, -1):
        if time.time() - start_time > deadline - 2:
            break
        n = sq * sq
        if n < 10000 or n > 1000000:
            continue
        pn_direct = _powerful_part_direct(n)
        pn1_direct = _powerful_part_direct(n + 1)
        val = pn_direct * pn1_direct
        if val > best_val:
            best_val = val
            best_n = n
        # Also check n-1
        if n > 10000:
            nm1 = n - 1
            pnm1_direct = _powerful_part_direct(nm1)
            val2 = pnm1_direct * pn_direct
            if val2 > best_val:
                best_val = val2
                best_n = nm1

    return int(best_n)
