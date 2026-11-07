"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 135.0  # safety margin before 150s kill

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

    def solve_one(n, edges, lists, seed_offset=0):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to sets for fast lookup
        list_sets = [set(lst) for lst in lists]
        
        # Degree for tiebreaking in MRV
        degree = [len(adj[v]) for v in range(n)]
        
        # Domains as sets for fast intersection
        domains = [set(lst) for lst in lists]
        assigned = [None] * n
        n_assigned = 0
        
        def backtrack():
            nonlocal n_assigned
            if time.time() > deadline:
                return False
            
            if n_assigned == n:
                return True
            
            # Find unassigned vertex with minimum domain size (MRV)
            # Tiebreak by higher degree
            min_v = -1
            min_size = float('inf')
            min_deg = -1
            for v in range(n):
                if assigned[v] is None:
                    size = len(domains[v])
                    if size < min_size or (size == min_size and degree[v] > min_deg):
                        min_size = size
                        min_v = v
                        min_deg = degree[v]
                        if size <= 1:
                            break
            
            if min_v == -1:
                return True  # All assigned
            
            if min_size == 0:
                return False  # No available colour
            
            # Try each colour in the domain
            for c in list(domains[min_v]):
                assigned[min_v] = c
                n_assigned += 1
                
                # Forward checking: remove c from neighbors' domains
                saved = {}
                consistent = True
                for u in adj[min_v]:
                    if assigned[u] is None:
                        if c in domains[u]:
                            saved[u] = domains[u].copy()
                            domains[u].discard(c)
                            if len(domains[u]) == 0:
                                consistent = False
                                break
                
                if consistent and backtrack():
                    return True
                
                # Undo
                for u, d in saved.items():
                    domains[u] = d
                assigned[min_v] = None
                n_assigned -= 1
            
            return False
        
        if backtrack():
            return assigned
        
        # Fallback: try a different approach with list-based domains
        # This handles edge cases where set operations might have issues
        domains2 = [list(lst) for lst in lists]
        assigned2 = [None] * n
        
        def backtrack2():
            if time.time() > deadline:
                return False
            
            # Find unassigned vertex with minimum domain size
            min_v = -1
            min_size = float('inf')
            for v in range(n):
                if assigned2[v] is None:
                    size = len(domains2[v])
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size <= 1:
                            break
            
            if min_v == -1:
                return True
            
            if min_size == 0:
                return False
            
            for c in list(domains2[min_v]):
                assigned2[min_v] = c
                
                saved = {}
                consistent = True
                for u in adj[min_v]:
                    if assigned2[u] is None:
                        if c in domains2[u]:
                            saved[u] = domains2[u][:]
                            domains2[u] = [x for x in domains2[u] if x != c]
                            if len(domains2[u]) == 0:
                                consistent = False
                                break
                
                if consistent and backtrack2():
                    return True
                
                for u, d in saved.items():
                    domains2[u] = d
                assigned2[min_v] = None
            
            return False
        
        if backtrack2():
            return assigned2
        
        return None

    out = []
    for idx, (splits, seed) in enumerate(((20, 701), (40, 702), (60, 703), (100, 704))):
        if time.time() > deadline - 5:
            break
        
        n, edges, lists = instance(splits, seed)
        result = solve_one(n, edges, lists, seed_offset=idx * 1000)
        
        if result is None:
            # Fallback: greedy with degree ordering
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            colour = [None] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        taken.add(colour[u])
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            result = colour
        
        # Verify and ensure types
        valid = True
        for v in range(n):
            if not isinstance(result[v], int) or result[v] not in lists[v]:
                valid = False
                break
        if valid:
            for u, v in edges:
                if result[u] == result[v]:
                    valid = False
                    break
        
        if not valid:
            # Last resort: first colour from list
            result = [int(lst[0]) for lst in lists]
        
        out.append([int(c) for c in result])
    
    # If we ran out of time, fill remaining with simple assignments
    while len(out) < 4:
        idx = len(out)
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[idx]
        n, edges, lists = instance(splits, seed)
        out.append([int(lst[0]) for lst in lists])
    
    return out
# EVOLVE-BLOCK-END
