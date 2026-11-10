import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    
    n = 18
    best_score = n + 1
    best_adj = None
    
    # Precompute indices for combinations? No, just use itertools.
    
    def get_k4s(adj):
        """Yield tuples (u,v,w,x) that form a K4."""
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] >> v) & 1:
                    continue
                for w in range(v+1, n):
                    if not ((adj[u] >> w) & 1 and (adj[v] >> w) & 1):
                        continue
                    for x in range(w+1, n):
                        if ((adj[u] >> x) & 1 and (adj[v] >> x) & 1 and 
                            (adj[w] >> x) & 1):
                            yield (u, v, w, x)

    def remove_k4s(adj):
        """Remove edges to make graph K4-free."""
        # Copy adj
        adj = adj[:]
        while True:
            if time.time() > deadline - 5.0:
                return adj
            # Find one K4
            k4 = None
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        continue
                    for w in range(v+1, n):
                        if not ((adj[u] >> w) & 1 and (adj[v] >> w) & 1):
                            continue
                        for x in range(w+1, n):
                            if ((adj[u] >> x) & 1 and (adj[v] >> x) & 1 and 
                                (adj[w] >> x) & 1):
                                k4 = (u, v, w, x)
                                break
                        if k4: break
                    if k4: break
                if k4: break
            
            if k4 is None:
                return adj
            
            # Remove a random edge from the K4
            edges = [
                (k4[0], k4[1]),
                (k4[0], k4[2]),
                (k4[0], k4[3]),
                (k4[1], k4[2]),
                (k4[1], k4[3]),
                (k4[2], k4[3])
            ]
            e_u, e_v = random.choice(edges)
            adj[e_u] &= ~(1 << e_v)
            adj[e_v] &= ~(1 << e_u)

    def max_triangle_free_size(adj):
        """Calculate the size of the largest triangle-free induced subgraph."""
        # Precompute common neighbors
        cn = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                cn[u][v] = adj[u] & adj[v]
        
        # Check sizes from n down to 1
        for k in range(n, 0, -1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for u in comb:
                    mask |= (1 << u)
                
                is_tf = True
                # Check all pairs in comb
                for i in range(len(comb)):
                    u = comb[i]
                    for j in range(i+1, len(comb)):
                        v = comb[j]
                        if (adj[u] >> v) & 1:
                            if cn[u][v] & mask:
                                is_tf = False
                                break
                    if not is_tf:
                        break
                if is_tf:
                    return k
        return 0

    # Search loop
    for seed in range(1000):
        if time.time() > deadline - 10.0:
            break
        
        random.seed(seed)
        
        # Generate random graph
        adj = [0] * n
        p = 0.6
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Make K4-free
        adj = remove_k4s(adj)
        
        # Calculate score
        score = max_triangle_free_size(adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
            # If we get a very good score, we might stop early?
            # Theoretical min for n=18?
            # If best_score <= 5, it's very good.
            if best_score <= 6:
                 # Continue a bit more to see if we can go lower, but don't wait too long
                 if time.time() > start_time + 30:
                     break

    if best_adj is None:
        # Fallback to seed
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
