import math
import time
from collections import deque

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
    num_primes = len(primes)
    
    # Precompute square-free kernel (bitmask) for each number in (210, max_value]
    sqf_bits = [0] * (max_value + 1)
    for n in range(211, max_value + 1):
        temp = n
        bits = 0
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                bits |= (1 << prime_to_idx[p])
        if temp > 1:
            bits |= (1 << prime_to_idx[temp])
        sqf_bits[n] = bits
    
    # Target: square-free part of 210 = 2*3*5*7
    target_bits = 0
    for p in [2, 3, 5, 7]:
        target_bits |= (1 << prime_to_idx[p])
    
    # BFS-like search: we want to find the minimal M such that there exists
    # a subset of {211, ..., M} whose product's square-free part equals target_bits.
    # Equivalently, in GF(2), the XOR of the sqf_bits of the selected numbers = target_bits.
    
    # We'll do a Dijkstra-like search where the "distance" is the max value used.
    # State: current XOR value (bitmask of primes)
    # Start: 0 (empty product)
    # Goal: target_bits
    # Edges: XOR with sqf_bits[n] for some n, but we want to minimize max(n)
    
    # Alternative approach: BFS over subsets is infeasible. Instead, we can think of it as:
    # For each M from 211 upward, check if target_bits is in the span of {sqf_bits[211], ..., sqf_bits[M]}
    # This is what the original code does. Let's keep that but optimize.
    
    # Actually, let's try a different approach: since we want to minimize max(selected),
    # we can process numbers in increasing order and maintain a basis. The first time
    # target_bits becomes reducible to 0, we have a solution. But we need to ensure
    # the solution uses at most max_size numbers.
    
    # Let's use the same incremental basis approach but with a twist: we track the
    # minimum max-value solution for each basis vector.
    
    # Actually, the simplest correct approach: for M = 211, 212, ..., check if
    # target is in span. The first M where this happens gives the minimal max.
    
    # But we need to be careful: the solution found might use more than max_size numbers.
    # If so, we need to continue searching for a valid solution with <= max_size numbers.
    
    basis = {}  # pivot -> (vec, combo)
    best_solution = None
    best_max = max_value + 1
    
    for M in range(211, max_value + 1):
        if time.time() > deadline:
            break
        
        v = sqf_bits[M]
        idx = M - 211
        comb = 1 << idx
        
        curr_v = v
        curr_comb = comb
        
        while curr_v != 0:
            hb = curr_v.bit_length() - 1
            if hb in basis:
                b_vec, b_combo = basis[hb]
                curr_v ^= b_vec
                curr_comb ^= b_combo
            else:
                basis[hb] = (curr_v, curr_comb)
                break
        
        # Check if target is in span
        t_curr = target_bits
        t_comb = 0
        
        while t_curr != 0:
            hb = t_curr.bit_length() - 1
            if hb in basis:
                b_vec, b_combo = basis[hb]
                t_curr ^= b_vec
                t_comb ^= b_combo
            else:
                break
        
        if t_curr == 0:
            sol_mask = t_comb
            sol_count = bin(sol_mask).count('1')
            
            if 1 <= sol_count <= max_size:
                result = []
                for i in range(M - 210):
                    if (sol_mask >> i) & 1:
                        result.append(211 + i)
                
                # Verify
                prod = base
                for x in result:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    best_solution = result
                    best_max = M
                    break
            # If too many numbers, continue searching
    
    if best_solution is not None:
        return list(best_solution)
    
    # Fallback
    return [840]
