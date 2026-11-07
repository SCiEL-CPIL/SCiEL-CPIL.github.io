"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    # Deadline guard: 150s limit, use 140s for safety
    deadline = time.time() + 140.0

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
        
        # Initialize domains
        domains = [set(lst) for lst in lists]
        colors = [None] * n
        
        # MRV Heuristic: Pick vertex with smallest domain
        def get_mrv_vertex(assigned):
            min_size = float('inf')
            min_vertex = -1
            for v in range(n):
                if assigned[v] is None:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        min_vertex = v
            return min_vertex

        # Forward Checking: Check if any neighbor's domain becomes empty
        def is_consistent(v, c):
            for u in adj[v]:
                if colors[u] is not None:
                    if colors[u] == c:
                        return False
            return True

        def backtrack():
            # Check time
            if time.time() > deadline:
                return False
            
            # Find unassigned vertex with MRV
            v = get_mrv_vertex(colors)
            if v == -1:
                return True # All assigned
            
            # Try each color in domain
            for c in sorted(domains[v]): # Sorted for determinism
                if not is_consistent(v, c):
                    continue
                
                # Assign
                colors[v] = c
                # Save state for backtracking (simple copy for small n)
                old_domains = [set(d) for d in domains]
                
                # Forward checking: remove c from neighbors' domains
                consistent = True
                for u in adj[v]:
                    if colors[u] is None:
                        if c in domains[u]:
                            domains[u].remove(c)
                            if len(domains[u]) == 0:
                                consistent = False
                                break
                
                if consistent:
                    if backtrack():
                        return True
                
                # Restore state
                for i in range(n):
                    domains[i] = old_domains[i]
                colors[v] = None
            
            return False

        if backtrack():
            return colors
        else:
            # Should not happen due to Alon-Tarsi, but return None if failed
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we can't finish properly.
            # However, the solver is fast. If this triggers, it's a bug or extreme slowness.
            # Return empty lists to avoid invalid type errors? No, must return list of lists of ints.
            # If we fail, the score is 0.
            # We will just break and return what we have, but the format must be strict.
            # Given the speed, this won't happen.
            break
            
        n, edges, lists = instance(splits, seed)
        coloring = solve_list_coloring(n, edges, lists)
        
        if coloring is None:
            # Fallback: If backtracking failed (shouldn't), try the greedy from seed?
            # Or just return an invalid coloring?
            # The task says "Any broken rule makes the whole result invalid".
            # So we must return a valid coloring.
            # Since it's guaranteed solvable, backtracking should work.
            # If it doesn't, it's a bug in my solver.
            # Let's add a simple greedy as a last resort just in case my backtracker has a bug.
            def greedy_fallback(n, edges, lists):
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                col = [None] * n
                for v in sorted(range(n), key=lambda u: -len(adj[u])):
                    taken = {col[u] for u in adj[v] if col[u] is not None}
                    free = [c for c in lists[v] if c not in taken]
                    if free:
                        col[v] = free[0]
                    else:
                        # This is the risky part. If this happens, the coloring is invalid.
                        # But we assume it won't happen for these instances.
                        col[v] = lists[v][0]
                return col
            coloring = greedy_fallback(n, edges, lists)
            
        # Ensure types are correct
        out.append([int(c) for c in coloring])
        
    return out
# EVOLVE-BLOCK-END
