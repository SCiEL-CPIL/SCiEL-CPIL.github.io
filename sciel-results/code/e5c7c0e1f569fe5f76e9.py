"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
sys.setrecursionlimit(10000)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110
    
    def make_disjoint_triangles(n_tri):
        edges = []
        for t in range(n_tri):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a,b], [a,c], [b,c]]
        return n_tri*3, edges
    
    best_n, best_edges = make_disjoint_triangles(10)
    
    def has_c4(n, adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if len(adj[u] & adj[v]) > 1:
                        return True
        return False
    
    def find_independent_set(n, adj, target_size):
        """Try to find independent set of size target_size. Return True if found."""
        # Use greedy coloring to get an upper bound, then backtracking
        # Simple backtracking with ordering
        remaining = list(range(n))
        
        def backtrack(rem, current):
            if len(current) >= target_size:
                return True
            if not rem:
                return False
            # Order by degree in induced subgraph
            rem_sorted = sorted(rem, key=lambda v: len(adj[v] & set(rem)))
            v = rem_sorted[0]
            # Include v
            new_rem = [x for x in rem if x != v and x not in adj[v]]
            if backtrack(new_rem, current + [v]):
                return True
            # Exclude v
            if backtrack([x for x in rem if x != v], current):
                return True
            return False
        
        return backtrack(remaining, [])
    
    def try_random_construction(target_n, seed, time_limit=10):
        rng = random.Random(seed)
        start = time.time()
        
        adj = [set() for _ in range(target_n)]
        
        # Phase 1: Add random edges while maintaining C4-free
        attempts = 0
        while attempts < target_n * target_n and time.time() - start < time_limit * 0.3:
            u, v = rng.sample(range(target_n), 2)
            if v not in adj[u] and len(adj[u] & adj[v]) <= 1:
                adj[u].add(v)
                adj[v].add(u)
            attempts += 1
        
        # Phase 2: Try to add more edges to reduce independence number
        while time.time() - start < time_limit * 0.7:
            added = False
            for u in range(target_n):
                for v in range(u+1, target_n):
                    if v not in adj[u] and len(adj[u] & adj[v]) <= 1:
                        adj[u].add(v)
                        adj[v].add(u)
                        added = True
            if not added:
                break
        
        # Check independence number
        if not find_independent_set(target_n, adj, 11):
            edges = []
            for u in range(target_n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return target_n, edges
        return 0, None
    
    # Try to find a 39-vertex construction
    for target_n in [39, 38, 37, 36]:
        for seed in range(100):
            if time.time() > deadline:
                break
            n, edges = try_random_construction(target_n, seed, time_limit=0.5)
            if n > best_n:
                best_n = n
                best_edges = edges
                if best_n >= 39:
                    return {'n': best_n, 'edges': best_edges}
    
    if best_edges is not None and best_n > 30:
        return {'n': best_n, 'edges': best_edges}
    
    return {'n': 30, 'edges': make_disjoint_triangles(10)[1]}
# EVOLVE-BLOCK-END
