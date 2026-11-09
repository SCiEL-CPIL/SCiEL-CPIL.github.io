import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    limit = 10210
    
    # Sieve primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute masks for all numbers from 211 to limit
    # Use a more efficient factorization approach
    mask_cache = [0] * (limit + 1)
    
    for n in range(2, limit + 1):
        temp = n
        mask = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask ^= (1 << prime_to_idx[p])
        if temp > 1:
            mask ^= (1 << prime_to_idx[temp])
        mask_cache[n] = mask
    
    target_mask = mask_cache[210]
    
    # RREF basis: basis[pivot] = (vector_mask, combo_mask)
    # Standard RREF: only reduce new vectors against existing basis
    # No need to update existing basis vectors (we just need span membership)
    basis = {}  # pivot_bit -> (vector_mask, combo_mask)
    
    # Track if target is in span incrementally
    # We'll check after each insertion
    
    for M in range(211, limit + 1):
        if time.time() > deadline:
            break
        
        idx = M - 211
        v = mask_cache[M]
        c = 1 << idx
        
        # Reduce v against existing basis
        while v != 0:
            pivot = v.bit_length() - 1
            if pivot in basis:
                b_mask, b_combo = basis[pivot]
                v ^= b_mask
                c ^= b_combo
            else:
                # Insert new basis vector
                basis[pivot] = (v, c)
                break
        
        # Check if target is in span
        curr_v = target_mask
        curr_c = 0
        in_span = True
        while curr_v != 0:
            pivot = curr_v.bit_length() - 1
            if pivot in basis:
                b_mask, b_combo = basis[pivot]
                curr_v ^= b_mask
                curr_c ^= b_combo
            else:
                in_span = False
                break
        
        if in_span:
            # Reconstruct solution
            sol_indices = []
            for i in range(idx + 1):
                if curr_c & (1 << i):
                    sol_indices.append(i + 211)
            
            if 1 <= len(sol_indices) <= 100:
                # Verify solution
                prod = 210
                for x in sol_indices:
                    prod *= x
                root = math.isqrt(prod)
                if root * root == prod:
                    return [int(x) for x in sol_indices]
    
    # Fallback
    return [224, 240]
