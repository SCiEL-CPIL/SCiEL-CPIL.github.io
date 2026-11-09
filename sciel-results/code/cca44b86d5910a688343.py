"""Improved program using segmented sieve for efficient p-smooth checking."""
import math
import time

# EVOLVE-BLOCK-START

def sieve_primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    time_limit = 110
    
    max_p_search = 5000000
    all_primes = sieve_primes(max_p_search)
    
    best_k = 0
    best_p = 2
    
    for idx_p, p in enumerate(all_primes):
        if time.time() - start_time > time_limit:
            break
        
        start = p * p + 1
        end = p * p + 200
        size = 200
        
        vals = [start + i for i in range(size)]
        non_one_count = size
        
        for q in all_primes:
            if q > p:
                break
            if q > end:
                break
            first = ((start + q - 1) // q) * q
            for j in range(first, end + 1, q):
                idx = j - start
                if vals[idx] == 1:
                    continue
                old_val = vals[idx]
                while vals[idx] % q == 0:
                    vals[idx] //= q
                if vals[idx] == 1 and old_val != 1:
                    non_one_count -= 1
                if non_one_count == 0:
                    break
            if non_one_count == 0:
                break
        
        k = 0
        for i in range(size):
            if vals[i] == 1:
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            if k == 200:
                break
    
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
