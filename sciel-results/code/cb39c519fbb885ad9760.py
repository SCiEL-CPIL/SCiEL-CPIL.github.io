import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    base = 210
    max_value = 10210
    max_size = 100
    
    # Sieve primes up to max_value
    sieve = [True] * (max_value + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_value**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_value + 1, i):
                sieve[j] = False
    primes = [i for i in range(max_value + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute square-free bitsets using SPF
    spf = list(range(max_value + 1))
    for i in range(2, int(max_value**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_value + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    sqf_bits = [0] * (max_value + 1)
    for n in range(211, max_value + 1):
        temp = n
        bits = 0
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                bits |= (1 << prime_to_idx[p])
        sqf_bits[n] = bits
        
    target_bits = 0
    for p in [2, 3, 5, 7]:
        target_bits |= (1 << prime_to_idx[p])

    # Incremental basis building with linear scan
    basis = {}
    combos = {}
    
    best_result = None
    for M in range(211, max_value + 1):
        if time.time() > deadline:
            break
        
        i = M - 211
        n = M
        v = sqf_bits[n]
        comb = 1 << i
        
        curr_v = v
        curr_comb = comb
        
        while curr_v != 0:
            hb = curr_v.bit_length() - 1
            if hb in basis:
                b_vec = basis[hb]
                b_combo = combos[hb]
                curr_v ^= b_vec
                curr_comb ^= b_combo
            else:
                basis[hb] = curr_v
                combos[hb] = curr_comb
                break
        
        # Check if target is in span
        t_curr = target_bits
        t_comb = 0
        
        while t_curr != 0:
            hb = t_curr.bit_length() - 1
            if hb in basis:
                b_vec = basis[hb]
                b_combo = combos[hb]
                t_curr ^= b_vec
                t_comb ^= b_combo
            else:
                break
        
        if t_curr == 0:
            sol_mask = t_comb
            sol_count = bin(sol_mask).count('1')
            if 1 <= sol_count <= max_size:
                result = []
                for j in range(M - 210):
                    if (sol_mask >> j) & 1:
                        result.append(211 + j)
                # Verify
                prod = base
                for x in result:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    best_result = result
                    break
    
    if best_result:
        return list(best_result)
        
    # Fallback
    return [840]
