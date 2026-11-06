import math
import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for 90s limit
    
    N = 256
    
    # 1. Pre-filter vertices where 2*v is a perfect square
    # These cannot be in any valid set because v+v = 2v would be a square.
    def is_square(n):
        if n < 0: return False
        r = int(math.isqrt(n))
        return r*r == n

    valid_vertices = []
    for i in range(1, N + 1):
        if not is_square(2 * i):
            valid_vertices.append(i)
            
    M = len(valid_vertices)
    if M == 0:
        return []
        
    # Map original value to index in valid_vertices
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 2. Build adjacency masks
    # adj[i] is a bitmask of neighbors of valid_vertices[i]
    adj = [0] * M
    
    # Precompute squares
    squares = set()
    k = 2
    while k*k <= 2 * N:
        squares.add(k*k)
        k += 1
        
    for i in range(M):
        vi = valid_vertices[i]
        mask = 0
        for j in range(M):
            if i == j: continue
            vj = valid_vertices[j]
            if (vi + vj) in squares:
                mask |= (1 << j)
        adj[i] = mask
        
    # 3. Greedy initialization for incumbent
    def greedy_mis():
        # Simple greedy on original indices
        selected = []
        used = 0
        indices = list(range(M))
        # Sort by degree ascending
        degrees_orig = [bin(adj[i]).count('1') for i in range(M)]
        indices.sort(key=lambda x: degrees_orig[x])
        
        for u in indices:
            if (adj[u] & used) == 0:
                selected.append(u)
                used |= (1 << u)
                
        return len(selected), used

    # 4. Reorder vertices by degree (descending) for B&B
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    order = list(range(M))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    new_idx = [0] * M
    old_idx = [0] * M
    for i, old in enumerate(order):
        new_idx[i] = old
        old_idx[old] = i
        
    # Rebuild adj in new order
    new_adj = [0] * M
    for i in range(M):
        old_i = new_idx[i]
        mask = 0
        for j in range(M):
            if i == j: continue
            old_j = new_idx[j]
            if (adj[old_i] >> old_j) & 1:
                mask |= (1 << j)
        new_adj[i] = mask
        
    new_degrees = [bin(new_adj[i]).count('1') for i in range(M)]
    
    # 5. Color Bound Function
    def color_bound_new(mask):
        if mask == 0:
            return 0
        colors = 0
        uncolored = mask
        while uncolored:
            colors += 1
            available = uncolored
            colored_mask = 0
            while available:
                lsb = available & -available
                v = lsb.bit_length() - 1
                available ^= lsb
                colored_mask |= lsb
                available &= ~new_adj[v]
            uncolored &= ~colored_mask
        return colors

    # 6. B&B
    best_mask = 0
    best_count = 0
    
    # Initial greedy on new graph
    def greedy_new():
        selected = 0
        used = 0
        idxs = list(range(M))
        idxs.sort(key=lambda x: new_degrees[x])
        for u in idxs:
            if (new_adj[u] & used) == 0:
                selected |= (1 << u)
                used |= (1 << u)
        return selected, bin(selected).count('1')
        
    init_mask, init_count = greedy_new()
    best_mask = init_mask
    best_count = init_count
    
    def expand(P, X):
        nonlocal best_mask, best_count
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if bin(X).count('1') > best_count:
                best_count = bin(X).count('1')
                best_mask = X
            return
            
        # Bound
        bound = color_bound_new(P)
        if bin(X).count('1') + bound <= best_count:
            return
            
        # Select a vertex from P
        lsb = P & -P
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        P1 = P & ~new_adj[v] & ~(1 << v)
        X1 = X | (1 << v)
        
        # Branch 2: Exclude v
        P2 = P & ~(1 << v)
        X2 = X
        
        expand(P1, X1)
        expand(P2, X2)
        
    full_mask = (1 << M) - 1
    expand(full_mask, 0)
    
    # Map best_mask back to original values
    result = []
    for i in range(M):
        if (best_mask >> i) & 1:
            orig_idx = new_idx[i]
            result.append(valid_vertices[orig_idx])
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
