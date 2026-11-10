import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    n = 40
    # Start with 8 disjoint K5s
    adj = np.zeros((n, n), dtype=np.uint8)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u, v] = 1
                adj[v, u] = 1
    
    def has_c6_with_edge(u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 through (u,v) means there's a path of length 4 from u to v:
        u-a-b-c-v with all distinct vertices.
        """
        # Find all length-4 paths from u to v
        # u ~ a ~ b ~ c ~ v
        # a must be neighbor of u (not v)
        # c must be neighbor of v (not u)
        # b must be neighbor of both a and c
        # a, b, c all distinct from u, v and each other
        
        # Get neighbors of u (excluding v since we're adding that edge)
        neighbors_u = [x for x in range(n) if adj[u, x] == 1 and x != v]
        neighbors_v = [x for x in range(n) if adj[v, x] == 1 and x != u]
        
        for a in neighbors_u:
            for c in neighbors_v:
                if a == c:
                    continue
                # Check if there exists b such that adj[a,b]=1 and adj[b,c]=1
                # b must be distinct from u, v, a, c
                # We can check: is there a common neighbor of a and c that's not u,v,a,c?
                common = adj[a] & adj[c]
                # Exclude u, v, a, c
                for b in range(n):
                    if common[b] == 1 and b != u and b != v and b != a and b != c:
                        return True
        return False
    
    def edge_count():
        return int(np.sum(adj) // 2)
    
    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    edges.append([u, v])
        return edges
    
    best_edges = get_edges()
    best_count = edge_count()
    
    # Try to add edges greedily
    # Generate candidate edges in a good order
    # Try all missing edges
    missing = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 0:
                missing.append((u, v))
    
    # Shuffle for diversity, but also try structured order
    import random
    random.seed(42)
    random.shuffle(missing)
    
    added = 0
    for u, v in missing:
        if time.time() > deadline:
            break
        if adj[u, v] == 1:
            continue
        if not has_c6_with_edge(u, v):
            adj[u, v] = 1
            adj[v, u] = 1
            added += 1
            current_count = edge_count()
            if current_count > best_count:
                best_count = current_count
                best_edges = get_edges()
    
    # Try multiple restarts with different orderings if time permits
    while time.time() < deadline - 5.0:
        # Reset to best known
        adj[:] = 0
        for e in best_edges:
            adj[e[0], e[1]] = 1
            adj[e[1], e[0]] = 1
        
        # Generate new missing list with different shuffle
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 0:
                    missing.append((u, v))
        random.shuffle(missing)
        
        improved = False
        for u, v in missing:
            if time.time() > deadline:
                break
            if adj[u, v] == 1:
                continue
            if not has_c6_with_edge(u, v):
                adj[u, v] = 1
                adj[v, u] = 1
                current_count = edge_count()
                if current_count > best_count:
                    best_count = current_count
                    best_edges = get_edges()
                    improved = True
                if time.time() > deadline:
                    break
        
        if not improved:
            break
    
    return best_edges
