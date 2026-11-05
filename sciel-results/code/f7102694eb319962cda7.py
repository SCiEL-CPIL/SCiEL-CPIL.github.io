import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    target_n = 46
    min_deg_required = target_n - 39  # 7
    
    import random
    
    def try_construct(seed):
        random.seed(seed)
        n = target_n
        # adj[i] is a bitmask of neighbors
        adj = [0] * n
        
        p = 7.5 / (n - 1)
        
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Remove C4s
        # A C4 exists if there are u, v such that popcount(adj[u] & adj[v]) >= 2
        # We remove edges until no such pair exists.
        
        max_iter = 500
        for _ in range(max_iter):
            found = False
            for u in range(n):
                if found: break
                for v in range(u+1, n):
                    common = adj[u] & adj[v]
                    # Check if common has >= 2 bits
                    if common & (common - 1):
                        # Find one bit to remove
                        # Let's remove edge (u, w) where w is a common neighbor
                        # w = index of lowest set bit in common
                        # w = (common & -common).bit_length() - 1
                        w = (common & -common).bit_length() - 1
                        
                        # Remove edge (u, w)
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        
                        found = True
                        break
            if not found:
                break
        
        # Check min degree
        min_deg = 0
        for u in range(n):
            d = bin(adj[u]).count('1')
            if d < min_deg:
                min_deg = d
                if min_deg < min_deg_required:
                    break
        
        if min_deg >= min_deg_required:
            edges = []
            for u in range(n):
                mask = adj[u]
                # Iterate over set bits
                v = 0
                while mask:
                    if mask & 1:
                        if u < v:
                            edges.append([u, v])
                    mask >>= 1
                    v += 1
            return {'n': n, 'edges': edges}
        return None

    for seed in range(500):
        if time.time() > deadline:
            break
        res = try_construct(seed)
        if res is not None:
            return res
            
    return {'n': 39, 'edges': []}
