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
            # This ensures n has exact exponent k for prime p
            max_m = MAX_VAL // power
            if max_m >= 1:
                # m ranges from 1 to max_m
                # We want m % p != 0
                # Instead of creating array of m and masking, we can use arange with step?
                # No, we need to skip multiples of p.
                
                # Method: Create array of all m, mask out multiples of p
                # This is O(N/p^k) memory and time.
                
                # Optimization: 
                # Indices are power, 2*power, ... max_m*power
                # We want to exclude 2*power if p=2? No, exclude m where m%p==0.
                
                # Vectorized mask approach:
                ms = np.arange(1, max_m + 1)
                mask = (ms % p) != 0
                valid_m = ms[mask]
                indices = valid_m * power
                P[indices] *= power
            
            power = next_power
            k += 1
    
    if time.time() - start_time > deadline - 5:
        # Not enough time for careful scan; return a reasonable default
        # But P is likely computed or partially computed.
        # If P is all 1s, max product is 1.
        # Let's just return 10000 if we are in trouble.
        return 10000
    
    # Compute products using int64 to avoid overflow and precision loss
    # P values are at most ~1e6, product at most ~1e12, fits in int64
    products = P[10000:1000001] * P[10001:1000002]
    
    # Find top candidates
    top_k = 100
    if len(products) < top_k:
        top_k = len(products)
    
    top_indices = np.argpartition(-products, top_k)[:top_k]
    top_indices = top_indices[np.argsort(-products[top_indices])]
    
    # Verify top candidates with direct factorization
    best_n = 10000
    best_val = 0
    
    for idx in top_indices:
        if time.time() - start_time > deadline - 2:
            break
        n = 10000 + idx
        pn_direct = _powerful_part_direct(n)
        pn1_direct = _powerful_part_direct(n + 1)
        val = pn_direct * pn1_direct
        
        # Cross-check with sieve
        pn_sieve = int(P[n])
        pn1_sieve = int(P[n + 1])
        
        if pn_sieve != pn_direct or pn1_sieve != pn1_direct:
            # Sieve bug detected; trust direct computation
            if val > best_val:
                best_val = val
                best_n = n
        else:
            if val > best_val:
                best_val = val
                best_n = n
    
    # Also check the overall argmax as a fallback
    max_idx = np.argmax(products)
    n_max = 10000 + max_idx
    pn_direct = _powerful_part_direct(n_max)
    pn1_direct = _powerful_part_direct(n_max + 1)
    val_max = pn_direct * pn1_direct
    if val_max > best_val:
        best_n = n_max
    
    return int(best_n)
