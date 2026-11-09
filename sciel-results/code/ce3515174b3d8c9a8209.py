"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of numbers j (1..N) such that gcd(i, j) > 1
    # We index bits 0..N-1 corresponding to numbers 1..N
    # bit k corresponds to number k+1
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and not prime_factors[i].isdisjoint(prime_factors[j]):
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask
    
    # Weights
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = i
    
    # Initial available mask: all numbers 1..N
    full_mask = (1 << N) - 1
    
    best_sum = 0
    best_set_mask = 0
    
    def greedy_solve():
        avail = full_mask
        s = 0
        mask_sel = 0
        while avail:
            m = avail
            best_idx = -1
            best_w = -1
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                w = weights[idx + 1]
                if w > best_w:
                    best_w = w
                    best_idx = idx
                m ^= lsb
            if best_idx == -1: break
            v = best_idx + 1
            v_bit = 1 << best_idx
            s += weights[v]
            mask_sel |= v_bit
            avail &= ~(conflict_mask[v] | v_bit)
        return s, mask_sel

    g_sum, g_mask = greedy_solve()
    best_sum = g_sum
    best_set_mask = g_mask

    def dfs(available_mask, current_sum, current_set_mask):
        nonlocal best_sum, best_set_mask
        
        if time.time() > deadline:
            return
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set_mask = current_set_mask
            return
        
        # Upper bound: current_sum + sum of weights of available nodes
        ub = current_sum
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ub += weights[idx + 1]
            m ^= lsb
            
        if ub <= best_sum:
            return
        
        # Select a vertex to branch on
        # Heuristic: Pick the vertex with the highest weight
        m = available_mask
        best_idx = -1
        best_w = -1
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            w = weights[idx + 1]
            if w > best_w:
                best_w = w
                best_idx = idx
            m ^= lsb
        
        if best_idx == -1:
            return
            
        v = best_idx + 1
        v_bit = 1 << best_idx
        
        # Branch 1: Include v
        new_mask_include = available_mask & ~(conflict_mask[v] | v_bit)
        dfs(new_mask_include, current_sum + weights[v], current_set_mask | v_bit)
        
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude v
        new_mask_exclude = available_mask & ~v_bit
        dfs(new_mask_exclude, current_sum, current_set_mask)
        
        if time.time() > deadline:
            return

    dfs(full_mask, 0, 0)
    
    # Convert best_set_mask to list of ints
    result = []
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
    
    return result
# EVOLVE-BLOCK-END
