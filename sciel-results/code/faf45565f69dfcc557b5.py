"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    import random
    from random import Random
    
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

    def solve_instance(n, edges, lists, deadline):
        """Solve a single list colouring instance using backtracking with constraint propagation."""
        if time.time() > deadline:
            return None
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try multiple ordering strategies and pick the first that works
        orderings = [
            # Degree-based: highest degree first
            lambda: sorted(range(n), key=lambda v: -len(adj[v])),
            # Reverse degree
            lambda: sorted(range(n), key=lambda v: len(adj[v])),
            # Random with seed
            lambda: list(range(n)),
        ]
        
        for order_func in orderings:
            if time.time() > deadline:
                return None
            
            order = order_func()
            color = [-1] * n
            domains = [list(lst) for lst in lists]
            
            # Forward checking backtracking
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                
                v = order[idx]
                # Try each colour in the domain
                for c in domains[v]:
                    # Check if colour is valid for all uncoloured neighbours
                    valid = True
                    for u in adj[v]:
                        if color[u] == c:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Assign
                    color[v] = c
                    # Remove c from domains of uncoloured neighbours
                    saved = {}
                    for u in adj[v]:
                        if color[u] == -1:
                            if c in domains[u]:
                                saved[u] = domains[u].copy()
                                domains[u].remove(c)
                                if not domains[u]:
                                    # Backtrack
                                    for uu, dd in saved.items():
                                        domains[uu] = dd
                                    color[v] = -1
                                    valid = False
                                    break
                            else:
                                pass
                        else:
                            if color[u] == c:
                                for uu, dd in saved.items():
                                    domains[uu] = dd
                                color[v] = -1
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Restore
                    for uu, dd in saved.items():
                        domains[uu] = dd
                    color[v] = -1
                
                return False
            
            # Reset state
            color = [-1] * n
            domains = [list(lst) for lst in lists]
            
            if backtrack(0):
                return color
        
        return None

    deadline = time.time() + 100  # 100 second safety margin
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is not None:
            out.append(result)
        else:
            # Fallback: greedy
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
    
    return out
# EVOLVE-BLOCK-END
