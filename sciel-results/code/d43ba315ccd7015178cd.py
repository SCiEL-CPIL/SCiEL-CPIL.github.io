"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)

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

    def solve_list_coloring(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Degrees for tie-breaking
        degrees = [len(adj[i]) for i in range(n)]
        
        # Domains as lists for efficiency, or sets? Sets are good for membership.
        # We will use sets for domains.
        domains = [set(lst) for lst in lists]
        
        # AC-3 Implementation
        def ac3(queue):
            while queue:
                u, v = queue.pop()
                # Check if domains[u] and domains[v] are consistent
                # We need to remove values from domains[u] that have no support in domains[v]
                # i.e., for each c in domains[u], if there is no c' in domains[v] such that c != c', then c is unsupported.
                # Since the constraint is !=, a value c in domains[u] is supported by domains[v] if domains[v] contains any value other than c.
                # This simplifies to: if len(domains[v]) == 1 and domains[v] == {c}, then c is unsupported in domains[u].
                # Wait, the constraint is "adjacent vertices must differ".
                # So c in domains[u] is valid if there exists c' in domains[v] such that c != c'.
                # This is false only if domains[v] is a subset of {c}, i.e., domains[v] == {c}.
                
                # So, we only need to remove c from domains[u] if domains[v] == {c}.
                # And symmetrically, remove c' from domains[v] if domains[u] == {c'}.
                
                # Optimization:
                if len(domains[v]) == 1:
                    val = next(iter(domains[v]))
                    if val in domains[u]:
                        domains[u].discard(val)
                        if not domains[u]:
                            return False
                        # Add all neighbors of u (except v) to queue
                        for w in adj[u]:
                            if w != v:
                                queue.append((w, u))
                if len(domains[u]) == 1:
                    val = next(iter(domains[u]))
                    if val in domains[v]:
                        domains[v].discard(val)
                        if not domains[v]:
                            return False
                        # Add all neighbors of v (except u) to queue
                        for w in adj[v]:
                            if w != u:
                                queue.append((v, w))
            return True

        # Initial AC-3
        initial_queue = []
        for u in range(n):
            for v in adj[u]:
                initial_queue.append((u, v))
        
        if not ac3(initial_queue):
            return [lists[v][0] for v in range(n)] # Fallback, shouldn't happen if colorable

        color = [-1] * n
        unassigned = [True] * n

        def backtrack():
            # MRV Heuristic: pick unassigned vertex with smallest domain
            min_size = n + 1
            min_v = -1
            for v in range(n):
                if unassigned[v]:
                    if len(domains[v]) < min_size:
                        min_size = len(domains[v])
                        min_v = v
                        if min_size <= 1:
                            break
            
            if min_v == -1:
                return True  # All assigned
            
            # Try each color in domain of min_v
            # Order colors by least constraining value? 
            # For !=, all colors are equivalent in terms of constraint propagation to neighbors 
            # unless a neighbor has domain size 1. 
            # We'll just iterate.
            for c in list(domains[min_v]):
                # Assign
                color[min_v] = c
                unassigned[min_v] = False
                
                # Save state for rollback
                old_domains = [set(d) for d in domains]
                
                # Run AC-3 starting with neighbors of min_v
                # Actually, we can just run AC-3 on the whole graph or just the affected part.
                # Given n is small (max 8 + 4*100 = 408), running AC-3 on the whole graph is cheap enough.
                # But we can optimize by only queuing edges incident to min_v.
                
                queue = []
                for u in adj[min_v]:
                    if unassigned[u]:
                        queue.append((u, min_v))
                        queue.append((min_v, u))
                    
                if ac3(queue):
                    if backtrack():
                        return True
                
                # Rollback
                for i in range(n):
                    domains[i] = old_domains[i]
                color[min_v] = -1
                unassigned[min_v] = True
            
            return False
        
        if backtrack():
            return [int(c) for c in color]
        else:
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_list_coloring(n, edges, lists)
        # Ensure all values are Python ints
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
