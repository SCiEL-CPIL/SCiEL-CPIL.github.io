import time
import random
import numpy as np
from itertools import combinations

def _has_c4(adj, n):
    """Check if graph has a C4. adj is a list of sets."""
    for u in range(n):
        for v in range(u+1, n):
            if v in adj[u]:
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return True
    return False

def _min_degree(adj, n):
    return min(len(adj[i]) for i in range(n))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_n = 39
    best_edges = []
    
    def try_construct(n, target_deg, seed):
        """Try to build C4-free graph on n vertices with min_degree >= target_deg using demand-driven greedy."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        
        def would_create_c4(u, v):
            """Check if adding edge (u,v) creates a C4."""
            # A C4 is formed if there is a path u-a-b-v of length 3.
            # Adding u-v closes it.
            # We need a in N(u), b in N(v), a!=b, a~b.
            # This is equivalent to:
            # Exists b in N(v) such that N(b) intersects N(u).
            # Note: If b is in N(u), then b is a common neighbor.
            # N(b) & N(u) would contain b? No, b not in N(b).
            # So if b in N(u), the intersection is N(b) \ {u} & N(u) \ {b}?
            # Actually, just check intersection of sets.
            
            nu = adj[u]
            for b in adj[v]:
                if adj[b] & nu:
                    return True
            return False
        
        # Demand-driven greedy
        while True:
            if time.time() > deadline:
                break
            
            # Find vertex with minimum degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= target_deg:
                break
            
            # Pick a vertex u with min degree (randomly among those with min degree)
            min_vertices = [i for i in range(n) if len(adj[i]) == min_deg]
            u = rng.choice(min_vertices)
            
            # Find candidates v to connect to u
            candidates = []
            for v in range(n):
                if v == u or v in adj[u]:
                    continue
                if not would_create_c4(u, v):
                    candidates.append(v)
            
            if not candidates:
                # Stuck
                break
            
            # Pick v with lowest degree among candidates to help balance
            v = min(candidates, key=lambda x: len(adj[x]))
            
            adj[u].add(v)
            adj[v].add(u)
        
        # Check if we achieved the target
        md = min(len(adj[i]) for i in range(n))
        if md >= target_deg:
            # Verify no C4
            if not _has_c4(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        return None
    
    # Try different n values from high to low
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        target_deg = n - 39
        if target_deg < 1:
            target_deg = 1
        
        # Try multiple random seeds
        found = False
        for seed in range(200):
            if time.time() > deadline:
                break
            
            edges = try_construct(n, target_deg, seed)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break
        
        if found:
            # Since we iterate from high n to low n, if we find a valid graph for n,
            # it is the best we can get for this iteration depth.
            # We can break the outer loop if we found a valid graph for the highest possible n?
            # No, we want to maximize n.
            # If we find a valid graph for 47, we are done.
            # If we find for 46, we return 46.
            # So we should break the outer loop immediately upon finding a valid graph?
            # Yes, because we are checking n in descending order.
            # If we find a valid graph for n, no smaller n is better.
            # And we already tried larger n and failed.
            break
        
        if time.time() > deadline:
            break
    
    # If we didn't find anything better, return the trivial solution
    if best_n <= 39:
        return {'n': 39, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
