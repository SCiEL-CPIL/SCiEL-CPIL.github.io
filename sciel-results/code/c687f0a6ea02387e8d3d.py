import random
import time

def construct():
    deadline = time.time() + 110  # Safety margin
    best_edges = []
    n = 40
    
    # Precompute all edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def would_create_c6(u, v, adj):
        # Check if adding (u, v) creates a C6.
        # This is equivalent to checking if there is a simple path of length 4 between u and v in the current graph.
        # Path: u - a - c - d - b - v
        # a in N(u), b in N(v)
        # c in N(a), d in N(b)
        # edge (c, d) exists
        # Vertices u, a, c, d, b, v are distinct.
        
        N_u = adj[u]
        N_v = adj[v]
        
        for a in N_u:
            if a == v: continue # Should not happen if (u,v) not in adj, but a is neighbor of u.
            for b in N_v:
                if b == u: continue
                if a == b: continue # Path length would be 2 (u-a-v), not 4.
                # We need a path a - c - d - b
                # c is neighbor of a, d is neighbor of b
                # and (c,d) is an edge.
                for c in adj[a]:
                    if c == u: continue
                    if c == b: continue # c cannot be b
                    if c == a: continue
                    for d in adj[b]:
                        if d == v: continue
                        if d == a: continue # d cannot be a
                        if d == c: continue
                        # Check distinctness of all 6
                        if c == v or d == u: continue
                        if c == u or d == v: continue # Redundant but safe
                        
                        # Check if (c, d) is an edge
                        if d in adj[c]:
                            return True
        return False

    def count_c6(adj):
        # Not strictly needed for greedy, but for validation if we want to be sure.
        # We rely on the incremental check.
        pass

    best_score = -1
    
    while time.time() < deadline:
        adj = [set() for _ in range(n)]
        edges = []
        
        # Randomize edge order
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not would_create_c6(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_score:
            best_score = len(edges)
            best_edges = edges[:]
            
        # Early stop if we hit a very high score?
        # ex(40, C6) is around 105. If we get > 120, we are doing great.
        if best_score > 130:
            break

    return best_edges
