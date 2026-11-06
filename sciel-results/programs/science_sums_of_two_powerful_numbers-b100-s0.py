import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    Powerful number: n = a^2 * b^3 where b is square-free.
    """
    if limit < 1:
        return []
    
    powerful = set()
    powerful.add(1)
    
    # b must be square-free and b^3 <= limit
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    if max_b < 1:
        max_b = 1

    # Check square-freeness using primes up to isqrt(max_b)
    sieve_limit = isqrt(max_b) + 1
    if sieve_limit < 2:
        primes = []
    else:
        is_prime = bytearray(b'\x01') * (sieve_limit + 1)
        is_prime[0] = 0
        if sieve_limit >= 1:
            is_prime[1] = 0
        for i in range(2, isqrt(sieve_limit) + 1):
            if is_prime[i]:
                for j in range(i * i, sieve_limit + 1, i):
                    is_prime[j] = 0
        primes = [i for i in range(2, sieve_limit + 1) if is_prime[i]]
        
    for b in range(1, max_b + 1):
        t = b
        is_sf = True
        for p in primes:
            pp = p * p
            if pp > t:
                break
            if t % pp == 0:
                is_sf = False
                break
        if not is_sf:
            continue
            
        b3 = b * b * b
        if b3 > limit:
            continue
            
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
                
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_N = 5
    best_pairs = [[1, 4]]
    best_count = 1
    
    # We will try a large limit. 
    # L = 5e8 is proven to work in inspiration 1.
    # Let's try L = 5e8.
    # If time permits, we could try larger, but memory for counts array is 2*L * 4 bytes.
    # 5e8 -> 1e9 * 4 = 4GB.
    # 2e8 -> 4e8 * 4 = 1.6GB.
    
    limits_to_try = [500_000_000, 300_000_000, 100_000_000]
    
    for limit in limits_to_try:
        try:
            if time.time() > deadline - 30:
                break
                
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 10:
                continue
                
            K = len(powerful)
            if K == 0:
                continue
                
            pow_arr = np.array(powerful, dtype=np.int64)
            max_sum = 2 * limit + 1
            
            # Check memory feasibility for counts array
            # int32 is 4 bytes.
            # If max_sum * 4 > available memory, we might crash.
            # Assuming 8GB+ RAM, 5e8 is okay (4GB).
            
            try:
                counts = np.zeros(max_sum, dtype=np.int32)
            except MemoryError:
                continue

            # Chunked bincount
            # K * C * 8 bytes for sums matrix.
            # If K=48000, C=1000, Mem = 384MB. Safe.
            chunk_size = 1000
            
            for i in range(0, K, chunk_size):
                if time.time() > deadline - 5:
                    break
                    
                chunk = pow_arr[i:i + chunk_size]
                # Compute outer sum: shape (K, chunk_size)
                # This creates a temporary array of size K * chunk_size
                sums = pow_arr[:, None] + chunk[None, :]
                
                # Flatten and bincount
                chunk_counts = np.bincount(sums.ravel(), minlength=max_sum)
                counts += chunk_counts
                
            if time.time() > deadline - 2:
                continue
                
            # Find best N
            # We want N >= 2
            # counts[0] and counts[1] are 0 or invalid for sum of two powerfuls >= 1+1=2?
            # Min powerful is 1. 1+1=2.
            # So we can just argmax.
            
            if len(counts) > 0:
                # Ensure we don't pick N < 2 if possible, though argmax will likely be large
                if counts[2:].max() > 0:
                     max_idx = np.argmax(counts[2:]) + 2
                else:
                     max_idx = np.argmax(counts)
                     
                current_best_N = int(max_idx)
                current_best_count = int(counts[current_best_N])
                
                if current_best_count > best_count:
                    best_count = current_best_count
                    best_N = current_best_N
                    # Reconstruct pairs
                    best_pairs = []
                    pow_set = set(powerful)
                    # a <= b => a <= N/2
                    half_N = best_N // 2
                    for a in powerful:
                        if a > half_N:
                            break
                        b = best_N - a
                        if b in pow_set:
                            best_pairs.append([a, b])
                    
        except MemoryError:
            continue
        except Exception as e:
            continue
            
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
