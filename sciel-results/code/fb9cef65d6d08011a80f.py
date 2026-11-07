"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    # Deadline guard: 140 seconds (safety margin before 150s kill)
    _start = time.time()
    _DEADLINE = _start + 140.0

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
        """
        Solve the list-coloring problem using forward-checking with backtracking
        and MRV (Minimum Remaining Values) heuristic.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains: each vertex can take any color from its list
        domains = [list(l) for l in lists]
        
        # Count of assigned vertices
        assigned = [False] * n
        color = [-1] * n  # -1 means unassigned
        
        def is_deadline_exceeded():
            return time.time() > _DEADLINE
        
        def backtrack():
            if is_deadline_exceeded():
                return False
            
            # Find unassigned vertex with minimum remaining values (MRV)
            min_size = float('inf')
            min_vertex = -1
            for v in range(n):
                if not assigned[v]:
                    size = len(domains[v])
                    if size == 0:
                        # Domain wipeout - failure
                        return False
                    if size < min_size:
                        min_size = size
                        min_vertex = v
                        if size == 1:
                            break  # Can't do better than 1
            
            if min_vertex == -1:
                # All vertices assigned
                return True
            
            # Try each color in the domain of min_vertex
            v = min_vertex
            # Heuristic: try colors that appear least often in neighbors' domains
            # (Least Constraining Value - LCV)
            color_counts = {}
            for c in domains[v]:
                count = 0
                for u in adj[v]:
                    if not assigned[u] and c in domains[u]:
                        count += 1
                color_counts[c] = count
            ordered_colors = sorted(domains[v], key=lambda c: color_counts.get(c, 0))
            
            for c in ordered_colors:
                # Assign color c to vertex v
                color[v] = c
                assigned[v] = True
                
                # Forward check: remove c from domains of unassigned neighbors
                # Track what we removed so we can undo
                removed = []
                valid = True
                for u in adj[v]:
                    if not assigned[u]:
                        if c in domains[u]:
                            domains[u].remove(c)
                            removed.append((u, c))
                            if len(domains[u]) == 0:
                                valid = False
                                break
                
                if valid:
                    if backtrack():
                        return True
                
                # Undo: restore c to domains of neighbors
                for u, rc in removed:
                    domains[u].append(rc)
                    domains[u].sort()
                
                # Undo assignment
                assigned[v] = False
                color[v] = -1
            
            return False
        
        if backtrack():
            # Convert to Python int and ensure list type
            result = [int(c) for c in color]
            return result
        else:
            # Should not happen per Alon-Tarsi, but return None as fallback
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > _DEADLINE - 5:
            # Not enough time for remaining instances; append empty/invalid
            out.append([])
            continue
        
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists)
        
        if result is not None:
            # Ensure all values are Python int
            out.append([int(c) for c in result])
        else:
            # Fallback: try greedy as last resort
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])
    
    return out
# EVOLVE-BLOCK-END
