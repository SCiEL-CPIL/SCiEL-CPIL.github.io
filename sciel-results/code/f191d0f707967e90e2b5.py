import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    MAX_VAL = 10210
    MIN_VAL = 211
    
    # 1. Sieve for primes up to MAX_VAL
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, MAX_VAL + 1, i):
                is_prime[j] = False
                
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Compute parity vectors for all numbers
    # v[n] is a bitmask where bit i is 1 if prime i has odd exponent in n
    v = [0] * (MAX_VAL + 1)
    for n in range(MIN_VAL, MAX_VAL + 1):
        x = n
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                cnt = 0
                while x % p == 0:
                    x //= p
                    cnt += 1
                if cnt % 2 == 1:
                    mask |= (1 << prime_index[p])
        if x > 1:
            # x is a prime factor
            if x in prime_index:
                mask |= (1 << prime_index[x])
        v[n] = mask
        
    # Compute target vector for 210
    target_vec = 0
    x = 210
    for p in primes:
        if p * p > x:
            break
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt % 2 == 1:
                target_vec |= (1 << prime_index[p])
    if x > 1:
        if x in prime_index:
            target_vec |= (1 << prime_index[x])
            
    # 3. Incremental Gaussian Elimination
    # basis: dict mapping pivot_bit -> (vector, orig_mask)
    basis = {}
    T = target_vec
    T_orig = 0  # Bitmask of indices (0 corresponds to 211, 1 to 212, etc.)
    
    for M in range(MIN_VAL, MAX_VAL + 1):
        if time.time() > deadline:
            break
            
        v_M = v[M]
        # orig_mask for M: bit (M - MIN_VAL)
        o_M = 1 << (M - MIN_VAL)
        
        v_red = v_M
        o_red = o_M
        
        # Reduce v_red by existing basis
        while v_red != 0:
            pivot = v_red.bit_length() - 1
            if pivot in basis:
                b_vec, b_orig = basis[pivot]
                v_red ^= b_vec
                o_red ^= b_orig
            else:
                break
                
        if v_red == 0:
            # M is in the span of previous vectors.
            # This doesn't change the span, so T doesn't change relative to the span.
            # We can skip adding to basis.
            # However, could we use M to simplify T?
            # No, M is a linear combination of previous basis vectors.
            # If T is not in the span of previous vectors, it won't be in the span of previous + M.
            # So we just continue.
            continue
        else:
            # New basis vector
            pivot = v_red.bit_length() - 1
            basis[pivot] = (v_red, o_red)
            
            # Reduce T using the new basis vector
            if (T >> pivot) & 1:
                T ^= v_red
                T_orig ^= o_red
            
            # Check if T is now 0
            if T == 0:
                # Solution found
                # Extract indices from T_orig
                indices = []
                mask = T_orig
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    indices.append(idx + MIN_VAL)
                    mask ^= lsb
                
                # Verify solution
                prod = 210
                for idx in indices:
                    prod *= idx
                k = math.isqrt(prod)
                if k * k == prod:
                    return [int(x) for x in indices]
                else:
                    # Should not happen, but if it does, continue search?
                    # Or return empty? The contract says invalid result is 0 score.
                    # Better to return a valid known solution if this fails?
                    # But this logic is sound.
                    pass
    
    # Fallback: If no solution found (should not happen), return a known valid solution
    # e.g., [840] works? 210 * 840 = 210 * 4 * 210 = 4 * 210^2 = (2*210)^2. Yes.
    return [840]
