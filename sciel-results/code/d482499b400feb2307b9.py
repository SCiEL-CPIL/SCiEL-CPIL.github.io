import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    def is_valid(n, adj):
        """
        Check if graph is triangle-free and has no independent set of size 10.
        adj: list of integers (bitmasks)
        """
        # 1. Triangle-free check
        for u in range(n):
            # For each neighbor v > u
            # Iterate through set bits in adj[u]
            mask_u = adj[u]
            # We only need to check v > u
            # To avoid duplicates, we can just check if (adj[u] & adj[v]) has any bits.
            # But we need to be careful with performance.
            # n=40 is small.
            pass # We'll do this in the search loop or a separate fast check.

    def has_triangle(n, adj):
        for u in range(n):
            mask_u = adj[u]
            # Iterate neighbors v > u
            # To iterate bits:
            m = mask_u
            while m:
                # Get lowest set bit
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if v <= u:
                    continue
                # Check if u and v share a common neighbor
                if adj[u] & adj[v]:
                    return True
        return False

    def find_independent_set(n, adj, size=10):
        """
        Find an independent set of size 'size'.
        Returns a list of vertices or None.
        """
        # Backtracking with bitmasks
        # candidates: bitmask of vertices that can be added
        # current: list of vertices in current independent set
        
        def backtrack(candidates, current):
            if len(current) == size:
                return current
            if not candidates:
                return None
            
            # Heuristic: pick vertex with min degree in candidates?
            # Or just pick the first one.
            # To speed up, we can order vertices.
            
            # Try to pick a vertex v from candidates
            # We can iterate through all candidates
            m = candidates
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                
                # Include v
                new_candidates = candidates & ~adj[v] & ~(1 << v)
                # Note: adj[v] contains v's neighbors. We want to remove v and its neighbors.
                # Actually, candidates are vertices NOT in current set and NOT neighbors of current set.
                # So new_candidates = candidates - {v} - N(v)
                # In bitmask: candidates & ~(1 << v) & ~adj[v]
                
                result = backtrack(new_candidates, current + [v])
                if result:
                    return result
            return None

        all_vertices = (1 << n) - 1
        return backtrack(all_vertices, [])

    def try_build(n, time_limit):
        """
        Try to build a valid graph on n vertices.
        """
        # Initialize random graph
        # We want it to be triangle-free.
        # Start with empty graph and add edges?
        # Or start with random and remove triangles?
        
        # Strategy: Start with a random graph, remove triangles, then local search.
        
        for restart in range(1000):
            if time.time() > start_time + time_limit:
                return None
                
            # Init adj as list of int bitmasks
            adj = [0] * n
            
            # Generate random edges
            # To ensure we don't start with too many triangles, maybe use low p?
            # Or generate bipartite?
            # Let's try random p=0.5 and clean up.
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            # Remove triangles
            changed = True
            while changed:
                changed = False
                for u in range(n):
                    mask_u = adj[u]
                    m = mask_u
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        m ^= lsb
                        if v <= u:
                            continue
                        if adj[u] & adj[v]:
                            # Triangle exists involving u, v, and some w in adj[u] & adj[v]
                            # Remove edge (u,v)
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            changed = True
                            break # Restart u loop? Or just continue?
                    # If we broke inner, we should probably restart outer or just continue
                    # It's safer to just let the while loop catch remaining triangles.
            
            # Local Search
            # Goal: alpha < 10
            for step in range(5000):
                if time.time() > start_time + time_limit:
                    return None
                
                # Check if valid
                # First, ensure no triangles (should be maintained, but verify if we add edges)
                # We will add edges carefully.
                
                indep_set = find_independent_set(n, adj, 10)
                if not indep_set:
                    # Success
                    edges = []
                    for u in range(n):
                        mask = adj[u]
                        m = mask
                        while m:
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            m ^= lsb
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                # Try to break independent set
                # Find a pair (u, v) in indep_set such that adding (u,v) doesn't create triangle
                added = False
                # Shuffle indep_set to avoid bias
                random.shuffle(indep_set)
                for i in range(len(indep_set)):
                    for j in range(i + 1, len(indep_set)):
                        u = indep_set[i]
                        v = indep_set[j]
                        # Check if edge already exists
                        if adj[u] & (1 << v):
                            continue
                        # Check if adding (u,v) creates triangle
                        # Triangle if (adj[u] & adj[v]) is not empty
                        if not (adj[u] & adj[v]):
                            # Safe to add
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added = True
                            break
                    if added:
                        break
                
                if not added:
                    # Stuck. Remove a random edge to allow future additions?
                    # Or restart?
                    # Removing an edge might increase alpha, but it allows new edge additions.
                    # Let's try removing an edge incident to one of the vertices in indep_set
                    # that is blocking the most potential edges?
                    # Simpler: Remove a random edge from the graph.
                    all_edges = []
                    for u in range(n):
                        mask = adj[u]
                        m = mask
                        while m:
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            m ^= lsb
                            if u < v:
                                all_edges.append((u, v))
                    
                    if all_edges:
                        u, v = random.choice(all_edges)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        break # Empty graph, alpha=n, stuck.

        return None

    # Try n=40
    result = try_build(40, 50)
    if result:
        return result

    # Try n=39
    if time.time() < deadline:
        result = try_build(39, 40)
        if result:
            return result

    # Try n=38
    if time.time() < deadline:
        result = try_build(38, 20)
        if result:
            return result

    # Fallback
    # K_{9,9} on 18 vertices
    n = 18
    edges = [[i, j] for i in range(9) for j in range(9, 18)]
    return {'n': n, 'edges': edges}
