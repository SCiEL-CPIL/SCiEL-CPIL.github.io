import random
import time

def construct():
    N = 20
    # Bitmasks for neighbors
    # N[i] is an integer where bit j is set if i is connected to j
    N_mask = [0] * N
    # M[i][j] is bitmask of common neighbors of i and j
    # M[i][j] = N[i] & N[j]
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_add(u, v):
        # Check triples containing u: {u, x, y}
        # Common neighbors: N'(u) & N[x] & N[y]
        # N'(u) = N[u] | (1 << v)
        # Intersection = (N[u] | (1 << v)) & N[x] & N[y]
        # = (N[u] & N[x] & N[y]) | ((1 << v) & N[x] & N[y])
        # Size = popcount(N[u] & N[x] & N[y]) + (1 if v in N[x] and v in N[y] else 0)
        
        new_u = N_mask[u] | (1 << v)
        
        for x in range(N):
            if x == u or x == v: continue
            for y in range(x+1, N):
                if y == u or y == v: continue
                
                # Count common neighbors of u, x, y
                # (new_u & N_mask[x] & N_mask[y])
                common = new_u & N_mask[x] & N_mask[y]
                if popcount(common) > 2:
                    return False
        
        # Check triples containing v: {v, x, y}
        new_v = N_mask[v] | (1 << u)
        
        for x in range(N):
            if x == u or x == v: continue
            for y in range(x+1, N):
                if y == u or y == v: continue
                
                common = new_v & N_mask[x] & N_mask[y]
                if popcount(common) > 2:
                    return False
                    
        return True

    def edge_count():
        return sum(popcount(N_mask[i]) for i in range(N)) // 2

    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    best_count = 0
    best_masks = None
    
    start_time = time.time()
    deadline = 110.0
    
    # Use a fixed seed for reproducibility? 
    # random.seed(42)
    
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        # Reset
        N_mask = [0] * N
        
        random.shuffle(all_edges)
        
        added = 0
        for u, v in all_edges:
            if N_mask[u] & (1 << v):
                continue
            if is_valid_add(u, v):
                N_mask[u] |= (1 << v)
                N_mask[v] |= (1 << u)
                added += 1
                
        if added > best_count:
            best_count = added
            best_masks = N_mask[:]
            
    if best_masks is None:
        return []
        
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_masks[i] & (1 << j):
                edges.append([i, j])
                
    return edges
