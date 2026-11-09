"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(100000)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute bitmask for each vertex's list
        list_masks = []
        for v in range(n):
            m = 0
            for c in lists[v]:
                m |= (1 << c)
            list_masks.append(m)
        
        # Precompute neighbor masks: for each vertex, a bitmask of its neighbors
        # Actually, we need to update domains when we assign a color to a vertex.
        # When we assign color c to vertex v, for each neighbor u, we remove c from u's domain.
        
        # We'll maintain:
        # domains[v] = bitmask of available colors for v
        # color[v] = -1 if uncolored, else color
        
        # State for backtracking:
        # We need to be able to undo changes to domains.
        # Since n <= 408 and depth <= 408, we can store a snapshot of domains at each level?
        # That would be O(n^2) memory in worst case, which is ~160k integers, fine.
        # But copying 408 integers at each of 408 levels is 408*408 = 166k ops, times branching...
        # Better to use a trail (undo log).
        
        # Let's use a trail-based approach.
        # trail: list of (vertex, old_domain)
        
        # We'll use a recursive function that takes:
        # - current domains (list of ints)
        # - current color assignment (list of ints)
        # - trail (list of (v, old_domain))
        
        # To avoid copying the entire domains list, we'll modify it in place and use the trail to undo.
        
        # Heuristic: MRV (Minimum Remaining Values)
        # Choose the uncolored vertex with the fewest available colors.
        # Tie-break: highest degree.
        
        # For efficiency, we can precompute degrees.
        degrees = [len(adj[v]) for v in range(n)]
        
        # We'll use a simple O(n) scan to find MRV vertex.
        # n is at most ~408, so this is fast.
        
        def backtrack(domains, color, trail):
            # Find uncolored vertex with minimum domain size
            min_count = 5  # Max possible is 3, so 5 is infinity
            best_v = -1
            best_degree = -1
            
            for v in range(n):
                if color[v] == -1:
                    cnt = domains[v].bit_count()
                    if cnt == 0:
                        return False
                    if cnt < min_count or (cnt == min_count and degrees[v] > best_degree):
                        min_count = cnt
                        best_v = v
                        best_degree = degrees[v]
            
            if best_v == -1:
                return True  # All vertices colored
            
            v = best_v
            # Get available colors for v
            avail_mask = domains[v]
            
            # Try each color in avail_mask
            # Order colors by least constraining value? For simplicity, try in order.
            # We can also try colors that appear in fewer neighbors' domains first (LCV heuristic).
            # For now, just iterate through bits.
            
            c = 0
            while avail_mask:
                if avail_mask & 1:
                    # Try color c
                    color[v] = c
                    bit = 1 << c
                    
                    # Save old domains for neighbors and update
                    saved = []
                    valid = True
                    for u in adj[v]:
                        if color[u] == -1:
                            old = domains[u]
                            new = old & ~bit
                            if new != old:
                                saved.append((u, old))
                                domains[u] = new
                                if new == 0:
                                    valid = False
                                    break
                    
                    if valid:
                        if backtrack(domains, color, trail):
                            return True
                    
                    # Undo
                    color[v] = -1
                    for u, old in saved:
                        domains[u] = old
                
                avail_mask >>= 1
                c += 1
            
            return False
        
        # Initialize
        domains = list(list_masks)
        color = [-1] * n
        trail = []
        
        if backtrack(domains, color, trail):
            return [int(c) for c in color]
        else:
            # Should not happen per Alon-Tarsi, but return something
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        col = solve_instance(n, edges, lists)
        out.append(col)
    
    # Ensure types are correct
    final_out = []
    for col_list in out:
        final_out.append([int(c) for c in col_list])
    
    return final_out
# EVOLVE-BLOCK-END
