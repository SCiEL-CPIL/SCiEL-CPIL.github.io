import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for overhead/return
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = [False] * len(is_prime[i*i::i])
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Segmented Sieve for Phi
    BLOCK_SIZE = 10000
    
    def compute_phi_block(start, size):
        vals = [start + i for i in range(size)]
        rem = [start + i for i in range(size)]
        
        max_val = start + size - 1
        sqrt_max = int(math.isqrt(max_val))
        
        for p in primes:
            if p > sqrt_max:
                break
            
            first_multiple = ((start + p - 1) // p) * p
            if first_multiple > max_val:
                continue
                
            for m in range(first_multiple, start + size, p):
                idx = m - start
                vals[idx] -= vals[idx] // p
                while rem[idx] % p == 0:
                    rem[idx] //= p
        
        for i in range(size):
            if rem[i] > 1:
                vals[i] -= vals[i] // rem[i]
                
        return vals

    best_k = 1
    best_s = 1
    
    MAX_SEARCH = 10**9 
    current_start = 1
    
    while current_start <= MAX_SEARCH:
        if time.time() > deadline:
            break
            
        actual_size = min(BLOCK_SIZE, MAX_SEARCH - current_start + 1)
        
        phi_vals = compute_phi_block(current_start, actual_size)
        
        seen = set()
        left = 0
        max_len_in_block = 0
        start_of_max = 0
        
        for right in range(actual_size):
            v = phi_vals[right]
            while v in seen:
                seen.remove(phi_vals[left])
                left += 1
            seen.add(v)
            curr_len = right - left + 1
            if curr_len > max_len_in_block:
                max_len_in_block = curr_len
                start_of_max = left
        
        if max_len_in_block > best_k:
            best_k = max_len_in_block
            best_s = current_start + start_of_max
            
        current_start += actual_size
        
    return {'start': best_s, 'length': best_k}
