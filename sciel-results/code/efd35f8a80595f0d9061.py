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

        # Convert lists to sets for faster intersection
        lists_sets = [set(lst) for lst in lists]

        # Greedy attempt first
        # Order by degree descending
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        colour_g = [-1] * n
        success = True
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour_g[u] != -1:
                    taken.add(colour_g[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour_g[v] = free[0]
            else:
                success = False
                break
        if success:
            return colour_g

        # Backtracking with MRV and Forward Checking
        # domains[v] is the set of available colours for v
        domains = [set(lst) for lst in lists]
        assigned = [False] * n
        colour = [-1] * n

        # We need to track the size of domains for MRV
        # Using a recursive function with domain propagation
        
        sys.setrecursionlimit(10000)

        def backtrack():
            # Find unassigned vertex with minimum domain size (MRV)
            min_size = float('inf')
            min_v = -1
            for v in range(n):
                if not assigned[v]:
                    size = len(domains[v])
                    if size == 0:
                        return False # Failure
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size == 1:
                            break # Can't get better than 1
            
            if min_v == -1:
                return True # All assigned

            v = min_v
            # Save original domain
            orig_domain = domains[v].copy()
            
            for c in list(orig_domain):
                # Try assigning c to v
                assigned[v] = True
                colour[v] = c
                domains[v] = {c}
                
                # Forward Check: update domains of neighbours
                consistent = True
                backup = []
                for u in adj[v]:
                    if assigned[u]:
                        if colour[u] == c:
                            consistent = False
                            break
                    else:
                        if c in domains[u]:
                            backup.append((u, domains[u].copy()))
                            domains[u].remove(c)
                            if not domains[u]:
                                consistent = False
                                break
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo
                assigned[v] = False
                colour[v] = -1
                domains[v] = orig_domain
                for u, dom in backup:
                    domains[u] = dom

            return False

        if backtrack():
            return colour

        # Fallback: should not happen
        return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_one(n, edges, lists)
        # Ensure all values are Python ints
        out.append([int(c) for c in colouring])
    return out
# EVOLVE-BLOCK-END
