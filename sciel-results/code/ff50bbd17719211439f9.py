import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Target n=39 first, then 38, etc.
    for n in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline:
            break
        result = _try_build(n, deadline - time.time())
        if result:
            return result
            
    # Fallback: Small valid graph
    # K_{9,9} on 18 vertices is triangle free, alpha=9.
    n = 18
    edges = [[i, j] for i in range(9) for j in range(9, 18)]
    return {'n': n, 'edges': edges}

def _try_build(n, time_limit):
    start_t = time.time()
    
    # Use bitmasks for speed
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge
    
    def _has_triangle(adj):
        for u in range(n):
            # Check pairs of neighbors of u
            # If two neighbors are connected, it's a triangle
            # We can just check if adj[u] & adj[v] is non-empty for v in adj[u]
            # But we need to be careful not to count the same triangle 3 times or check self-loops (none)
            # Simple check:
            neighbors_u = adj[u]
            # Iterate through set bits in neighbors_u
            # For each v, check if adj[v] intersects neighbors_u
            # Actually, if we just check if any edge exists within the neighborhood of u, it's a triangle.
            # Let's just iterate u, v in neighbors_u, and check if adj[v] has any bit in neighbors_u other than u?
            # No, simpler:
            for v in range(n):
                if (adj[u] >> v) & 1:
                    # Check if u and v share a neighbor
                    # common = adj[u] & adj[v]
                    # If common is not empty, there is a triangle.
                    # But common might include u or v? No, no self loops.
                    # So if adj[u] & adj[v] != 0, there is a triangle involving u, v, and some w.
                    if adj[u] & adj[v]:
                        return True
        return False

    def _find_indep_set(adj, size):
        # Backtracking with bitmasks
        # candidates: bitmask of available vertices
        # current: list of vertices in current independent set
        
        def backtrack(candidates, current):
            if len(current) == size:
                return current
            if not candidates:
                return None
            
            # Heuristic: pick a vertex from candidates
            # To prune fast, we can pick a vertex with high degree in the subgraph induced by candidates?
            # Or just pick the first one.
            # Let's pick the lowest set bit for determinism or random for diversity?
            # Random helps escape local minima.
            
            # Get list of candidates
            # Optimization: if popcount(candidates) < size - len(current), return None
            # Python 3.10+ has int.bit_count(). Let's assume modern Python.
            if candidates.bit_count() < size - len(current):
                return None
                
            # Pick a vertex
            # Randomly select a bit
            # Or just pick the first one. Let's try picking the one with min degree in candidates to prune?
            # For now, let's just pick the first set bit for speed, and rely on multiple restarts.
            # Actually, picking a random vertex is better for finding *a* set if one exists.
            
            # Let's get a random candidate
            # Efficient way to pick random set bit:
            # mask = candidates
            # shift = random.randint(0, n-1)
            # This is not uniform.
            
            # Let's just iterate.
            # To make it faster, we can order candidates by degree.
            # But that's expensive.
            
            # Let's try a simple recursive step:
            # Pick v.
            # Option 1: Include v. New candidates = candidates & ~(adj[v] | (1<<v))
            # Option 2: Exclude v. New candidates = candidates & ~(1<<v)
            
            # Branching factor 2. Depth 10. 2^10 = 1024 paths. Very fast.
            
            # Let's pick v as the lowest set bit for now.
            v = (candidates & -candidates).bit_length() - 1
            
            # Include v
            new_cand_include = candidates & ~(adj[v] | (1 << v))
            res = backtrack(new_cand_include, current + [v])
            if res:
                return res
            
            # Exclude v
            new_cand_exclude = candidates & ~(1 << v)
            res = backtrack(new_cand_exclude, current)
            if res:
                return res
                
            return None

        all_vertices = (1 << n) - 1
        return backtrack(all_vertices, [])

    def _remove_triangle_edge(adj):
        # Find a triangle and remove one edge
        # Return True if an edge was removed
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    if common:
                        # Found triangle u, v, w
                        w = (common & -common).bit_length() - 1
                        # Remove edge (u, v)
                        mask = ~(1 << v)
                        adj[u] &= mask
                        mask = ~(1 << u)
                        adj[v] &= mask
                        return True
        return False

    def _add_safe_edge(adj, u, v):
        # Check if adding (u,v) creates a triangle
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    # Main loop
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() > start_t + time_limit:
            return None
            
        # Initialize
        adj = [0] * n
        # Random sparse graph
        p = 0.1
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        # Repair Loop
        for step in range(5000):
            if time.time() > start_t + time_limit:
                return None
                
            # 1. Remove Triangles
            if _has_triangle(adj):
                _remove_triangle_edge(adj)
                continue
                
            # 2. Check Independent Set
            indep_set = _find_indep_set(adj, 10)
            if not indep_set:
                # Success
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
            # 3. Break Independent Set
            # Try to add an edge between two vertices in indep_set
            added = False
            # Shuffle to try different pairs
            random.shuffle(indep_set)
            for i in range(len(indep_set)):
                for j in range(i+1, len(indep_set)):
                    u = indep_set[i]
                    v = indep_set[j]
                    if not ((adj[u] >> v) & 1): # Not already edge
                        if _add_safe_edge(adj, u, v):
                            added = True
                            break
                if added:
                    break
                    
            if not added:
                # Stuck. Try removing a random edge to change structure
                all_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            all_edges.append((u, v))
                if all_edges:
                    u, v = random.choice(all_edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    break
                    
    return None
