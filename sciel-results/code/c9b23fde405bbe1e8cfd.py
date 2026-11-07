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
        """
        Solves a single list colouring instance using backtracking 
        with dynamic MRV (Minimum Remaining Values) and LCV (Least Constraining Value) heuristics.
        Optimized to use a list of unassigned vertices for faster MRV selection.
        """
        if time.time() > deadline:
            return None
        
        # Build adjacency lists
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        colour = [-1] * n
        domains = [list(lst) for lst in lists]
        
        # List of unassigned vertices for efficient MRV scanning
        unassigned = list(range(n))
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            if not unassigned:
                return True
            
            # 1. Select Variable: MRV (Minimum Remaining Values)
            # Among unassigned vertices, pick the one with the smallest domain.
            # Tie-breaker: highest degree (most constrained by neighbors).
            min_size = float('inf')
            best_v = -1
            best_deg = -1
            
            for v in unassigned:
                size = len(domains[v])
                if size < min_size:
                    min_size = size
                    best_v = v
                    best_deg = len(adj[v])
                    if min_size <= 1: # Can't do better than 1
                        break
                elif size == min_size and len(adj[v]) > best_deg:
                    best_v = v
                    best_deg = len(adj[v])
            
            if best_v == -1:
                return True # Should not happen if unassigned is not empty
            
            v = best_v
            dom = domains[v]
            
            if not dom:
                return False # Domain wipeout
            
            # 2. Select Value: LCV (Least Constraining Value)
            # Try values that rule out the fewest choices for unassigned neighbors.
            lcv_order = []
            for c in dom:
                constraint_count = 0
                for u in adj[v]:
                    if colour[u] == -1 and c in domains[u]:
                        constraint_count += 1
                lcv_order.append((constraint_count, c))
            
            # Sort by least constraining first
            lcv_order.sort()
            
            for _, c in lcv_order:
                # Check if color c is valid with respect to already colored neighbors
                valid = True
                for u in adj[v]:
                    if colour[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign color c to v
                colour[v] = c
                # Remove v from unassigned list
                unassigned.remove(v)
                
                # Propagate: Remove c from domains of unassigned neighbors
                saved_domains = []
                feasible = True
                for u in adj[v]:
                    if colour[u] == -1:
                        if c in domains[u]:
                            old_dom = domains[u]
                            new_dom = [x for x in old_dom if x != c]
                            saved_domains.append((u, old_dom))
                            domains[u] = new_dom
                            if not new_dom:
                                feasible = False
                                break
                
                if not feasible:
                    # Undo propagation
                    for u, old_dom in saved_domains:
                        domains[u] = old_dom
                    colour[v] = -1
                    unassigned.append(v)
                    continue
                
                # Recurse
                if backtrack():
                    return True
                
                # Undo
                for u, old_dom in saved_domains:
                    domains[u] = old_dom
                colour[v] = -1
                unassigned.append(v)
            
            return False
        
        if backtrack():
            # Ensure types are correct (Python int)
            return [int(c) for c in colour]
        
        return None

    start = time.time()
    deadline = start + 110  # 110s safety margin out of 150s limit
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is not None:
            out.append(result)
        else:
            # Fallback: Greedy (should not trigger for these instances if solver is correct)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            # Order by degree descending
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                # Find a color in lists[v] not in taken
                chosen = 0
                for c in lists[v]:
                    if c not in taken:
                        chosen = c
                        break
                colour[v] = chosen
            out.append([int(c) for c in colour])
    
    return out
# EVOLVE-BLOCK-END
