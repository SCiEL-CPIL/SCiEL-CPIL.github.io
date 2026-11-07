"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

def construct():
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

    def solve_one(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initialize available colors as sets for fast modification
        avail = [set(lst) for lst in lists]
        color = [-1] * n

        # Increase recursion limit to handle deep backtracking if necessary
        sys.setrecursionlimit(10000)

        def backtrack():
            # Find uncolored vertex with Minimum Remaining Values (MRV)
            min_val = n + 1
            v_min = -1
            for i in range(n):
                if color[i] == -1:
                    len_avail = len(avail[i])
                    if len_avail < min_val:
                        min_val = len_avail
                        v_min = i
                        if len_avail == 0:
                            return False
                        if len_avail == 1:
                            break # Can't get better than 1

            if v_min == -1:
                return True # All vertices colored

            # Try each available color for v_min
            # Sorting colors might help with determinism or specific heuristics, 
            # but any order works for finding *a* solution.
            for c in list(avail[v_min]):
                # Assign color c to v_min
                color[v_min] = c
                
                # Forward Checking: Remove c from neighbors' available sets
                # We need to restore these if backtracking
                changes = []
                valid = True
                for u in adj[v_min]:
                    if color[u] == -1 and c in avail[u]:
                        avail[u].remove(c)
                        changes.append(u)
                        if len(avail[u]) == 0:
                            valid = False
                            break

                if valid:
                    if backtrack():
                        return True

                # Undo changes
                for u in changes:
                    avail[u].add(c)
                
                color[v_min] = -1

            return False

        if backtrack():
            return color
        
        # Fallback (should not happen for valid instances per Alon-Tarsi)
        return [-1] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_one(n, edges, lists)
        # Ensure all values are Python ints
        out.append([int(c) for c in colouring])
    return out
# EVOLVE-BLOCK-END
