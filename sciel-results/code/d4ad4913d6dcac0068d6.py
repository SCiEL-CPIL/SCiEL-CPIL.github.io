"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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

    def solve_instance(n, edges, lists, deadline):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [set(l) for l in lists]
        
        # Forward checking + backtracking
        assignment = [-1] * n
        unassigned = set(range(n))
        
        def mrv_select(unassigned_set, domains):
            """Select unassigned vertex with minimum domain size (MRV), tie-break by degree."""
            best = -1
            best_size = float('inf')
            best_deg = -1
            for v in unassigned_set:
                sz = len(domains[v])
                deg = len(adj[v])
                if sz < best_size or (sz == best_size and deg > best_deg):
                    best_size = sz
                    best_deg = deg
                    best = v
            return best
        
        def forward_check(v, color, unassigned_set, domains):
            """Remove color from neighbors' domains. Returns (success, new_domains, removed_info)."""
            new_domains = [d.copy() for d in domains]
            removed = {}  # neighbor -> (color, old_size)
            for u in adj[v]:
                if u in unassigned_set:
                    if color in new_domains[u]:
                        removed[u] = (color, len(new_domains[u]))
                        new_domains[u].discard(color)
                        if len(new_domains[u]) == 0:
                            return False, new_domains, removed
            return True, new_domains, removed
        
        def backtrack(unassigned_set, domains, assignment):
            if time.time() > deadline:
                return None
            
            if not unassigned_set:
                return assignment[:]
            
            v = mrv_select(unassigned_set, domains)
            if v == -1:
                return None
            
            unassigned_set.remove(v)
            
            # Try each color in domain
            for color in sorted(domains[v]):
                assignment[v] = color
                success, new_domains, removed = forward_check(v, color, unassigned_set, domains)
                if success:
                    result = backtrack(unassigned_set, new_domains, assignment)
                    if result is not None:
                        return result
                # Restore
                for u, (c, old_size) in removed.items():
                    domains[u].add(c)
            
            unassigned_set.add(v)
            assignment[v] = -1
            return None
        
        result = backtrack(set(range(n)), domains, assignment)
        return result
    
    # Deadline: 140s to be safe (evaluator kills at 150s)
    start_time = time.time()
    deadline = start_time + 140.0
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Check if we're running out of time
        if time.time() > deadline - 5.0:
            # Not enough time for another instance, return what we have
            # But we need to return 4 lists, so use greedy fallback for remaining
            break
        
        result = solve_instance(n, edges, lists, deadline)
        
        if result is not None and all(isinstance(c, int) and not isinstance(c, bool) for c in result):
            out.append(result)
        else:
            # Fallback: greedy
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
    
    # If we broke early, fill remaining with greedy
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        out.append(colour)
    
    # Ensure all are Python ints in lists
    final_out = []
    for colouring in out:
        final_out.append([int(c) for c in colouring])
    
    return final_out
# EVOLVE-BLOCK-END
