import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    N = 512
    
    # Precompute compatibility bitmasks
    # compat[i] is a bitmask where bit j is 1 if lcm(i, j) <= N
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                mask |= (1 << j)
        compat[i] = mask
        
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_mask(mask):
        # Check if the set represented by mask is a clique
        # For each i in mask, mask must be a subset of compat[i]
        # i.e., mask & ~compat[i] == 0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            i = lsb.bit_length() - 1
            if mask & ~compat[i]:
                return False
            m ^= lsb
        return True

    def complete(mask):
        # Greedily add all compatible elements
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline - 1.0:
                    return mask
                bit = 1 << x
                if mask & bit:
                    continue
                # Check if x is compatible with all elements in mask
                if mask & ~compat[x]:
                    continue
                mask |= bit
                changed = True
        return mask

    def local_search(mask):
        mask = complete(mask)
        best_mask = mask
        best_size = popcount(mask)
        improved = True
        rng = random.Random(42)
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            # 1-for-1 swaps
            # Try removing each element u and adding each element v
            m = mask
            elements = []
            while m:
                lsb = m & -m
                elements.append(lsb.bit_length() - 1)
                m ^= lsb
                
            for u in elements:
                if time.time() > deadline - 2.0:
                    break
                mask_minus_u = mask & ~(1 << u)
                # Try adding any v not in mask
                # Optimization: Only try v that are compatible with mask_minus_u
                # v is compatible with mask_minus_u if mask_minus_u & ~compat[v] == 0
                # We can iterate v from 1 to N
                for v in range(1, N + 1):
                    if time.time() > deadline - 2.0:
                        break
                    if v == u:
                        continue
                    if mask & (1 << v):
                        continue
                    if mask_minus_u & ~compat[v]:
                        continue
                    # Try the swap
                    new_mask_base = mask_minus_u | (1 << v)
                    new_mask = complete(new_mask_base)
                    new_size = popcount(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        # Restart element list for next iteration if we improved significantly?
                        # Or just continue with current mask
                        break # Break inner loop to restart outer loop with new mask? 
                        # Actually, breaking inner loop and continuing outer loop is fine, 
                        # but 'elements' is stale. Let's just break both.
            
            if improved:
                continue
                
            # 1-for-2 swaps
            # Remove u, try to add two elements
            m = mask
            elements = []
            while m:
                lsb = m & -m
                elements.append(lsb.bit_length() - 1)
                m ^= lsb
                
            for u in elements:
                if time.time() > deadline - 2.0:
                    break
                mask_minus_u = mask & ~(1 << u)
                added = 0
                new_mask = mask_minus_u
                # Greedily add up to 2 elements
                for v in range(1, N + 1):
                    if time.time() > deadline - 2.0:
                        break
                    if added >= 2:
                        break
                    if new_mask & (1 << v):
                        continue
                    if new_mask & ~compat[v]:
                        continue
                    new_mask |= (1 << v)
                    added += 1
                
                if added >= 2:
                    # Complete the new mask
                    new_mask = complete(new_mask)
                    new_size = popcount(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        break
                        
        return best_mask

    # Generate multiple initial candidates
    candidates = []
    
    # 1. Divisors of highly composite numbers
    for M in [360, 420, 480, 504, 512, 240, 180, 120]:
        divs = []
        for i in range(1, M + 1):
            if M % i == 0:
                divs.append(i)
        mask = 0
        for d in divs:
            mask |= (1 << d)
        candidates.append(mask)
        
    # 2. Greedy from small to large
    mask = 0
    for i in range(1, N + 1):
        if mask & ~compat[i] == 0:
            mask |= (1 << i)
    candidates.append(mask)
    
    # 3. Greedy from large to small
    mask = 0
    for i in range(N, 0, -1):
        if mask & ~compat[i] == 0:
            mask |= (1 << i)
    candidates.append(mask)
    
    # 4. Random restarts
    rng = random.Random(123)
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        mask = 0
        # Random subset of [1, 50] to start
        for i in range(1, 51):
            if rng.random() < 0.5:
                mask |= (1 << i)
        # Ensure validity of initial mask? 
        # It might not be valid. We need to fix it or start with valid.
        # Let's just start with {1} and add randomly
        mask = 1 << 1
        order = list(range(2, N + 1))
        rng.shuffle(order)
        for x in order:
            if mask & ~compat[x] == 0:
                mask |= (1 << x)
        candidates.append(mask)

    best_overall_mask = 0
    best_overall_size = 0
    
    for mask in candidates:
        if time.time() > deadline - 3.0:
            break
        # Ensure the candidate is valid? 
        # If not, complete() might fail or produce invalid set if logic is wrong.
        # But complete() only adds if compatible. 
        # However, if initial mask is invalid, complete() doesn't fix invalid pairs, it just adds.
        # So we must ensure initial masks are valid cliques.
        # Divisors of M are always valid cliques.
        # Greedy constructions are valid cliques by construction.
        # Random restarts: We built them by adding only compatible elements, so they are valid.
        
        optimized_mask = local_search(mask)
        size = popcount(optimized_mask)
        if size > best_overall_size:
            best_overall_size = size
            best_overall_mask = optimized_mask
            
    # Convert best mask to list
    result = []
    m = best_overall_mask
    while m:
        lsb = m & -m
        result.append(lsb.bit_length() - 1)
        m ^= lsb
        
    if not result:
        return list(range(1, 23))
        
    return sorted(result)

# EVOLVE-BLOCK-END
