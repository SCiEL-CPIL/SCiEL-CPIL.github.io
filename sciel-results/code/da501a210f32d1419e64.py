import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    
    # 1. Compute compatibility and degrees
    compat_mask = [0] * (n + 1)
    degrees = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) != 0:
                compat_mask[i] |= (1 << j)
                compat_mask[j] |= (1 << i)
                degrees[i] += 1
                degrees[j] += 1
                
    # 2. Sort vertices by degree descending
    # We want high degree vertices to have high indices in the new mapping
    # so that "highest bit" picks the highest degree vertex.
    vertices = list(range(1, n + 1))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map old index to new index (1-based for bitmask ease, or 0-based?)
    # Let's use 0-based for the new indices to make bit operations slightly cleaner?
    # No, 1-based is fine.
    # new_index[v] = position in sorted list (0 to n-1)
    # We will use bits 0 to n-1.
    new_index = [0] * (n + 1)
    for i, v in enumerate(vertices):
        new_index[v] = i
        
    # Rebuild compat_mask with new indices
    new_compat_mask = [0] * n
    for i in range(1, n + 1):
        old_mask = compat_mask[i]
        new_mask = 0
        # Iterate through bits set in old_mask
        m = old_mask
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > 0: # j is old index
                new_mask |= (1 << new_index[j])
            m -= lsb
        new_compat_mask[new_index[i]] = new_mask
        
    # 3. Greedy Initialization on original indices
    S = set(range(1, n + 1, 2))
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    for x in range(2, n + 1, 2):
        if (compat_mask[x] & S_mask) == S_mask:
            S.add(x)
            S_mask |= (1 << x)
            
    best_size = len(S)
    best_mask_original = S_mask
    
    # Convert best_mask_original to new index mask
    best_mask_new = 0
    m = best_mask_original
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        if i > 0:
            best_mask_new |= (1 << new_index[i])
        m -= lsb
        
    # 4. B&B Search on new indices
    # P is a bitmask of available vertices (0 to n-1)
    
    def greedy_color_bound(P):
        colors = 0
        U = P
        while U:
            colors += 1
            Q = U
            color_class = 0
            while Q:
                # Pick vertex with highest degree in Q? 
                # For simplicity, pick lowest bit. 
                # But since we ordered by degree, highest degree has highest index.
                # So we should pick the HIGHEST bit to prioritize high degree?
                # In Tomita, the ordering is used for the bound.
                # Let's pick the highest bit in Q.
                v = Q.bit_length() - 1
                color_class |= (1 << v)
                Q &= ~new_compat_mask[v]
                Q &= ~(1 << v)
            U &= ~color_class
        return colors

    best_size_val = best_size
    best_mask_val = best_mask_new
    
    def search(P, bound, path_mask):
        nonlocal best_size_val, best_mask_val, deadline, start_time
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if bound > best_size_val:
                best_size_val = bound
                best_mask_val = path_mask
            return
            
        if bound + P.bit_count() <= best_size_val:
            return
            
        ub = greedy_color_bound(P)
        if bound + ub <= best_size_val:
            return
            
        # Pick vertex with highest degree (highest index in our sorted order)
        v = P.bit_length() - 1
        
        # Branch 1: Include v
        new_P = P & new_compat_mask[v] & ~(1 << v)
        search(new_P, bound + 1, path_mask | (1 << v))
        
        # Branch 2: Exclude v
        P = P & ~(1 << v)
        search(P, bound, path_mask)
        
    all_mask_new = (1 << n) - 1
    search(all_mask_new, 0, 0)
    
    # 5. Map back to original indices
    result = []
    m = best_mask_val
    while m:
        lsb = m & -m
        new_i = lsb.bit_length() - 1
        # Find original index
        # We need inverse mapping
        # We can just iterate 1..n
        # But we have new_index.
        # Let's create inverse_new_index
        # Actually, we can just loop 1..n and check if new_index[i] is in best_mask_val
        pass
        
    # Simpler mapping back
    result = []
    for i in range(1, n + 1):
        if best_mask_val & (1 << new_index[i]):
            result.append(i)
            
    return result
