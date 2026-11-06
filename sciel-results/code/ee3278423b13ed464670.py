import random
import time
import math

def construct():
    # Sieve of Eratosthenes up to 10^6
    limit = 1000000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    def compute_phi_block(start, size):
        # Returns list of phi values for start, start+1, ..., start+size-1
        phi_vals = [start + i for i in range(size)]
        rem_vals = [start + i for i in range(size)]
        
        for p in primes:
            if p * p > start + size - 1:
                break
            # Find first multiple of p in [start, start+size-1]
            first_multiple = ((start + p - 1) // p) * p
            if first_multiple > start + size - 1:
                continue
            for j in range(first_multiple - start, size, p):
                if rem_vals[j] % p == 0:
                    phi_vals[j] -= phi_vals[j] // p
                    while rem_vals[j] % p == 0:
                        rem_vals[j] //= p
        
        # Handle remaining large prime factors
        for i in range(size):
            if rem_vals[i] > 1:
                phi_vals[i] -= phi_vals[i] // rem_vals[i]
                
        return phi_vals

    best_k = 1
    best_s = 1
    start_time = time.time()
    time_limit = 115
    
    # Phase 1: Sequential Scan using Segmented Sieve
    # Scan up to 10^7 (10 million)
    # This covers a much larger range than the previous 10^5
    block_size = 10000
    max_scan_limit = 10000000
    
    s = 1
    while s < max_scan_limit and time.time() - start_time < time_limit * 0.5:
        # Check bounds
        current_block_size = block_size
        if s + current_block_size > max_scan_limit:
            current_block_size = max_scan_limit - s + 1
            if current_block_size <= 0:
                break
                
        phi_vals = compute_phi_block(s, current_block_size)
        
        # Find longest run of distinct values in this block
        # Using sliding window technique
        left = 0
        window = {}
        max_len_in_block = 0
        max_start_in_block = 0
        
        for right in range(len(phi_vals)):
            val = phi_vals[right]
            if val in window:
                left = max(left, window[val] + 1)
            window[val] = right
            current_len = right - left + 1
            if current_len > max_len_in_block:
                max_len_in_block = current_len
                max_start_in_block = left
                if max_len_in_block > best_k:
                    best_k = max_len_in_block
                    best_s = s + max_start_in_block
                    if best_k == 200:
                        return {'start': best_s, 'length': best_k}
        
        s += current_block_size

    # Phase 2: Random Search
    while time.time() - start_time < time_limit:
        # Pick random start between 10^7 and 10^12
        log_s = random.uniform(7, 12)
        s = int(10 ** log_s)
        
        # Ensure s is within bounds
        if s < 1 or s + 200 > 10**12:
            continue
        
        seen = set()
        k = 0
        while k < 200:
            v = phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
    
    return {'start': best_s, 'length': best_k}
