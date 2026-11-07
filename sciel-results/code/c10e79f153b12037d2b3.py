"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains as sets for fast removal
        domains = [set(lst) for lst in lists]
        
        # Check initial validity
        for d in domains:
            if not d:
                return None

        # Recursive backtracking with MRV and Forward Checking
        def backtrack(domains, assigned):
            # Find unassigned vertex with MRV
            min_size = float('inf')
            min_v = -1
            for v in range(n):
                if v not in assigned:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size == 1:
                            break
            if min_v == -1:
                return assigned # Success

            # Order values: Least Constraining Value (LCV) is optional, 
            # but for small domains, simple iteration is fine.
            for color in sorted(domains[min_v]):
                new_assigned = assigned | {min_v}
                new_domains = [set(d) for d in domains]
                new_domains[min_v] = {color}
                
                valid = True
                for neighbor in adj[min_v]:
                    if neighbor in new_assigned:
                        # Neighbor is already assigned, so its domain is a singleton {c_neigh}
                        # If color == c_neigh, conflict
                        if color in new_domains[neighbor]:
                            valid = False
                            break
                    else:
                        if color in new_domains[neighbor]:
                            new_domains[neighbor].remove(color)
                            if not new_domains[neighbor]:
                                valid = False
                                break
                
                if valid:
                    res = backtrack(new_domains, new_assigned)
                    if res is not None:
                        return res
            
            return None

        assignment = backtrack(domains, set())
        if assignment is None:
            return None
        
        return [assignment[v] for v in range(n)]

    out = []
    start_time = time.time()
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Time check
        if time.time() - start_time > 145:
            # Emergency fallback: return a list of zeros (will likely score 0 for this instance, 
            # but keeps structure valid). 
            # However, we should avoid this.
            coloring = [0] * n
        else:
            coloring = solve_instance(n, edges, lists)
            if coloring is None:
                # Should not happen
                coloring = [0] * n
        
        out.append([int(c) for c in coloring])
        
    return out
# EVOLVE-BLOCK-END
