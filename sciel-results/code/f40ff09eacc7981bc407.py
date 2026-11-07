import math
import time

def construct():
    # Primes up to 10210
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute parity bitmasks for all numbers in range
    # bitmask is an int where bit i is 1 if prime[i] has odd exponent
    bitmasks = {}
    for n in range(211, limit + 1):
        mask = 0
        x = n
        for i, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                count = 0
                while x % p == 0:
                    x //= p
                    count += 1
                if count % 2 == 1:
                    mask |= (1 << i)
        if x > 1:
            # x is a prime
            idx = prime_to_idx[x]
            mask |= (1 << idx)
        bitmasks[n] = mask
        
    # Target bitmask for 210
    target_mask = 0
    x = 210
    for i, p in enumerate(primes):
        if p * p > x:
            break
        if x % p == 0:
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count % 2 == 1:
                target_mask |= (1 << i)
    if x > 1:
        idx = prime_to_idx[x]
        target_mask |= (1 << idx)
        
    # Gaussian Elimination State
    # basis: dict mapping pivot_index -> (vector, combination_mask)
    # We maintain reduced row echelon form for correctness
    basis = {}
    
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin for the 90s kill
    
    for M in range(211, limit + 1):
        # Check deadline
        if time.time() > deadline:
            return [840]
            
        v = bitmasks[M]
        
        # Try to reduce v using current basis
        # Process pivots in decreasing order for correct reduction
        u = v
        c = 1 << (M - 211)
        
        # Eliminate u using basis, processing highest pivots first
        for pivot in sorted(basis.keys(), reverse=True):
            if (u >> pivot) & 1:
                b_vec, b_comb = basis[pivot]
                u ^= b_vec
                c ^= b_comb
        
        if u != 0:
            # Find pivot for u (highest set bit)
            pivot = u.bit_length() - 1
            if pivot not in basis:
                # Add new vector and eliminate its pivot from existing basis vectors
                # to maintain reduced row echelon form
                for p2 in list(basis.keys()):
                    if (u >> p2) & 1:
                        # This shouldn't happen if we reduced correctly, but just in case
                        pass
                # Actually, we need to eliminate this new pivot from existing vectors
                for p2 in list(basis.keys()):
                    if (basis[p2][0] >> pivot) & 1:
                        b_vec2, b_comb2 = basis[p2]
                        basis[p2] = (b_vec2 ^ u, b_comb2 ^ c)
                basis[pivot] = (u, c)
                
        # Check if target is in span
        # Reduce target_mask using basis, processing highest pivots first
        t_u = target_mask
        t_c = 0
        for pivot in sorted(basis.keys(), reverse=True):
            if (t_u >> pivot) & 1:
                b_vec, b_comb = basis[pivot]
                t_u ^= b_vec
                t_c ^= b_comb
        
        if t_u == 0:
            # Solution found
            selected = []
            for i in range(M - 211 + 1):
                if (t_c >> i) & 1:
                    selected.append(211 + i)
            
            if len(selected) > 100:
                continue
                
            # Verify with isqrt
            prod = 1
            for x in selected:
                prod *= x
            prod *= 210
            s = math.isqrt(prod)
            if s * s == prod:
                return list(selected)
            else:
                pass
                
    # Fallback
    return [840]
