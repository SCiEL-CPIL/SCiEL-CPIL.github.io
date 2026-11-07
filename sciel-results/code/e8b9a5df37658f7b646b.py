"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import deque

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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initial domains
        domains = [set(l) for l in lists]
        colour = [-1] * n
        
        # Heuristic: Minimum Remaining Values (MRV)
        # We pick the uncoloured vertex with the smallest domain.
        # Tie-breaker: Highest degree (Least Constraining Value heuristic for variable selection)
        
        def find_mrv_vertex():
            min_size = float('inf')
            best_v = -1
            best_deg = -1
            for v in range(n):
                if colour[v] == -1:
                    size = len(domains[v])
                    if size < min_size or (size == min_size and len(adj[v]) > best_deg):
                        min_size = size
                        best_v = v
                        best_deg = len(adj[v])
            return best_v

        def backtrack():
            # Check if all coloured
            # Optimization: if any domain is empty, fail
            v = find_mrv_vertex()
            if v == -1:
                return True # All vertices coloured
            
            if not domains[v]:
                return False
            
            # Try each colour in the domain
            # Sort colours by Least Constraining Value (LCV): 
            # Prefer colours that remove the fewest options from neighbours
            # For simplicity and speed on small graphs, we can just iterate, 
            # but sorting by LCV can help.
            # Given small size (3 colours), overhead of sorting might not be worth it, 
            # but let's try simple iteration first.
            
            for c in sorted(domains[v]):
                # Check consistency with already coloured neighbours
                conflict = False
                for u in adj[v]:
                    if colour[u] != -1 and colour[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Assign
                colour[v] = c
                domains[v].clear() # Mark as used/assigned
                
                # Forward Checking
                saved_states = []
                valid = True
                for u in adj[v]:
                    if colour[u] == -1:
                        if c in domains[u]:
                            domains[u].remove(c)
                            saved_states.append((u, c))
                            if not domains[u]:
                                valid = False
                                break
                
                if valid and backtrack():
                    return True
                
                # Undo
                colour[v] = -1
                domains[v].add(c)
                for u, c_val in saved_states:
                    domains[u].add(c_val)
                    
            return False

        if backtrack():
            return [int(c) for c in colour]
        else:
            # Fallback (should not happen)
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))
    return out
# EVOLVE-BLOCK-END
