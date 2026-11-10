import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        # adj[i] is a bitmask of neighbors of i
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return bin(x).count('1')

    def solve_instance(n, p, seed, num_restarts=1000):
        adj_template = generate_graph(n, p, seed)
        best_cliques = None
        best_count = float('inf')
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            adj = [x for x in adj_template]
            cliques = []
            
            # Copy adj for modification
            # We need to track which edges are removed.
            # Removing a clique C means for all u,v in C, u!=v, remove edge (u,v).
            
            while True:
                # Find vertex with max degree
                max_deg = -1
                v = -1
                for u in range(n):
                    d = popcount(adj[u])
                    if d > max_deg:
                        max_deg = d
                        v = u
                        # Add small random tie-breaker?
                        # if d == max_deg and random.random() < 0.5: ...
                
                if max_deg == 0:
                    break
                
                # Build clique containing v
                C = [v]
                # Candidates are neighbors of v
                candidates = adj[v]
                
                # Greedily add vertices to C
                while candidates:
                    # Find w in candidates that has max common neighbors in candidates
                    best_w = -1
                    best_score = -1
                    # Iterate through bits in candidates
                    c = candidates
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        # Score: number of neighbors of w that are also in candidates
                        score = popcount(adj[w] & candidates)
                        # Add random tie breaker
                        score += random.random() * 0.1
                        if score > best_score:
                            best_score = score
                            best_w = w
                        c ^= lsb
                    
                    if best_w == -1:
                        break
                    
                    C.append(best_w)
                    # Update candidates: must be neighbor of best_w AND in current candidates
                    candidates &= adj[best_w]
                
                # Remove edges in clique C
                for u in C:
                    for v2 in C:
                        if u != v2:
                            adj[u] &= ~(1 << v2)
                            # adj[v2] &= ~(1 << u) # Symmetric, but we update both to be safe? 
                            # Actually, adj is symmetric. If we clear u from v2's adj, we must clear v2 from u's adj.
                            # The loop above does this for all pairs.
                
                cliques.append(C)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = [list(c) for c in cliques]
                
        return best_cliques

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
