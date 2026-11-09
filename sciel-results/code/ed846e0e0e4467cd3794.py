import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    
    # Helper for GF(4)
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    def mul(x, y):
        return mul_table[x][y]
    def add(x, y):
        return x ^ y
        
    # Generate PG(2,4) points
    points = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x==0 and y==0 and z==0: continue
                if x != 0:
                    inv = [0, 1, 3, 2][x]
                    p = (1, mul(y, inv), mul(z, inv))
                elif y != 0:
                    inv = [0, 1, 3, 2][y]
                    p = (0, 1, mul(z, inv))
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
                    
    # Generate edges of polarity graph
    pg_edges = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            p = points[i]
            q = points[j]
            val = add(mul(p[0], q[2]), add(mul(p[1], q[1]), mul(p[2], q[0])))
            if val == 0:
                pg_edges.append((i, j))
                
    # Select best 20-vertex subgraph by removing one vertex
    best_initial = []
    for remove in range(21):
        current = []
        for u, v in pg_edges:
            if u == remove or v == remove:
                continue
            nu = u if u < remove else u - 1
            nv = v if v < remove else v - 1
            current.append((nu, nv))
        if len(current) > len(best_initial):
            best_initial = current
            
    # Initialize adjacency from best_initial
    adj = [0] * n
    initial_edges = []
    for u, v in best_initial:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        initial_edges.append([u, v])
        
    best_edges = initial_edges
    best_count = len(initial_edges)
    
    # Get all possible edges not in initial
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    existing = set((min(u,v), max(u,v)) for u,v in best_initial)
    remaining_edges = [e for e in all_edges if (min(e[0],e[1]), max(e[0],e[1])) not in existing]
    
    range_n = range(n)
    
    while time.time() < deadline:
        # Shuffle remaining edges to try in random order
        random.shuffle(remaining_edges)
        
        current_adj = adj[:]
        current_edges = best_edges[:]
        
        # Greedy add
        for u, v in remaining_edges:
            if current_adj[u] & (1 << v):
                continue # Already present? Should not happen if we removed them, but safe
            
            # Check validity
            valid = True
            
            # Check triples containing u
            # Optimization: only check if potential K3,3 can form
            # Common neighbors of u and a, b must be >= 3
            for a in range_n:
                if a == u: continue
                for b in range_n:
                    if b <= a or b == u: continue
                    common = current_adj[u] & current_adj[a] & current_adj[b]
                    if common.bit_count() >= 3:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Check triples containing v
                for a in range_n:
                    if a == v: continue
                    for b in range_n:
                        if b <= a or b == v: continue
                        common = current_adj[v] & current_adj[a] & current_adj[b]
                        if common.bit_count() >= 3:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.append([u, v])
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            # Update remaining edges? 
            # The set of edges that were NOT added might change?
            # No, we are adding to the best found so far.
            # The "remaining" list should be updated to exclude newly added edges.
            new_existing = set((min(u,v), max(u,v)) for u,v in current_edges)
            remaining_edges = [e for e in all_edges if (min(e[0],e[1]), max(e[0],e[1])) not in new_existing]
            adj = current_adj[:]

    return best_edges
