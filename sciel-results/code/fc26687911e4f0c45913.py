import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph for this n
        # We'll try a few random graphs
        # p = 0.5 is a good start, but maybe others are better?
        # Let's try a few p values
        
        found = False
        for p in [0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8]:
            if time.time() > deadline:
                break
                
            # Try multiple random graphs for this p
            for _ in range(50):  # Max 50 tries per p
                if time.time() > deadline:
                    break
                
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < p:
                            edges.append([u, v])
                
                if is_valid(n, edges):
                    return {'n': n, 'edges': edges}
        
        # If we didn't find one for this n, try smaller n
        # We don't return here, we just continue to next n
    
    # Fallback: return a small valid graph
    # n=5 is always valid
    return {'n': 5, 'edges': []}

def is_valid(n, edges):
    neighbors = [0] * n
    for u, v in edges:
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)
    
    # Check all subsets of size >= 6
    # Optimization: if n < 6, always valid
    if n < 6:
        return True
        
    full_mask = (1 << n) - 1
    
    # Iterate through all masks
    # To speed up, we can skip masks with < 6 bits
    # But iterating 1 to 2^n is fine for n<=22
    
    for mask in range(1, 1 << n):
        if mask.bit_count() < 6:
            continue
        
        # Check if induced subgraph on 'mask' is regular
        # Get first vertex
        lsb = mask & -mask
        u = lsb.bit_length() - 1
        target_deg = (neighbors[u] & mask).bit_count()
        
        # Check other vertices
        m = mask ^ lsb
        is_reg = True
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (neighbors[v] & mask).bit_count()
            if deg != target_deg:
                is_reg = False
                break
            m ^= lsb
        
        if is_reg:
            return False
            
    return True
