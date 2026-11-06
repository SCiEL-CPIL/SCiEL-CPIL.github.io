import time
import numpy as np
from math import isqrt, gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Step 1: Generate powerful numbers up to L
    L = 2 * 10**8
    powerful_set = set()
    max_b = int(L ** (1.0/3.0)) + 1
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            powerful_set.add(n)
    
    powerful_list = sorted(powerful_set)
    
    # Step 2: Sieve primes up to sqrt(L) for fast is_powerful
    limit_primes = isqrt(L) + 1
    sieve = bytearray(b'\x01') * (limit_primes + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit_primes) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, limit_primes + 1) if sieve[i]]
    
    # Step 3: Fast is_powerful with caching
    _cache = {}
    def is_powerful(n):
        if n in _cache:
            return _cache[n]
        if n == 1:
            _cache[n] = True
            return True
        x = n
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                cnt = 0
                while x % p == 0:
                    x //= p
                    cnt += 1
                if cnt < 2:
                    _cache[n] = False
                    return False
        if x > 1:
            # x is prime with exponent 1
            _cache[n] = False
            return False
        _cache[n] = True
        return True
    
    # Step 4: Target-centred search
    # Use LCM of small powerful numbers as structure
    # LCM(4,8,9,16,25,27,36,49,64,81,100) = 6350400
    # But that's too sparse. Use smaller: LCM(4,9,25) = 900
    # Or LCM(4,8,9,25) = 1800
    # Let's use multiple LCMs to cover different structures
    
    LCM_options = [900, 1800, 3600, 7200, 14400, 28800, 57600, 115200]
    
    best_N = 2
    best_count = 0
    best_pairs = []
    
    p_set = powerful_set
    p_list = powerful_list
    is_pow = is_powerful
    
    # For each LCM, check multiples up to a reasonable limit
    # Time budget: ~100s for search
    # Each N check: iterate p_list (~100k for L=2e8), but break when a > N
    # For large N, this is expensive. Limit N range.
    
    max_N_check = 5 * 10**8
    
    for lcm_val in LCM_options:
        if time.time() > deadline - 20:
            break
        num_multiples = max_N_check // lcm_val
        # Limit total work: aim for ~20s per LCM
        # Each N check ~ len(p_list) operations worst case
        # p_list has ~100k elements for L=2e8
        # 20s / 100k ~ 200 N's per LCM? That's too few.
        # Optimize: only check a up to min(N, L) since b = N-a must be >= 1
        # And use early break
        
        for k in range(1, num_multiples + 1):
            if time.time() > deadline - 15:
                break
            N = k * lcm_val
            if N > max_N_check:
                break
            if N <= best_N and best_count > 0:
                # Still check, might have same count but different structure
                pass
            
            count = 0
            # Iterate a in p_list, a <= N
            # Optimize: a must be <= N, and b = N-a >= 1
            for a in p_list:
                if a > N:
                    break
                b = N - a
                if b < 1:
                    break
                if b in p_set:
                    count += 1
                else:
                    if is_pow(b):
                        count += 1
            
            # Convert ordered to unordered
            if N % 2 == 0:
                half = N // 2
                if half in p_set or is_pow(half):
                    unordered = (count + 1) // 2
                else:
                    unordered = count // 2
            else:
                unordered = count // 2
            
            if unordered > best_count:
                best_count = unordered
                best_N = N
                # Generate pairs for this N
                best_pairs = []
                for a in p_list:
                    if a > N // 2:
                        break
                    b = N - a
                    if b < 1:
                        break
                    if b in p_set:
                        best_pairs.append([a, b])
                    else:
                        if is_pow(b):
                            best_pairs.append([a, b])
                best_pairs.sort()
    
    # Fallback: if no good N found, use a simple one
    if best_count == 0:
        best_N = 5
        best_pairs = [[1, 4]]
    
    return {'N': best_N, 'pairs': best_pairs}
