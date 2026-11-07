"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

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

    def solve_instance(n, edges, lists, deadline, rng):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        degree = [len(adj[i]) for i in range(n)]
        
        # Static ordering: sort by degree descending, then by index for determinism
        order = sorted(range(n), key=lambda v: (-degree[v], v))
        
        # Domains as lists (small, size 3)
        domains = [list(lst) for lst in lists]
        assignment = [-1] * n
        unassigned = [True] * n
        unassigned_count = n
        
        # Precompute for each vertex in order, which earlier vertices are neighbors
        # For forward checking, when assigning v, we remove its color from unassigned neighbors
        
        def try_color(v, c, domains, assignment, unassigned, unassigned_count):
            """Try assigning color c to vertex v with forward checking.
            Returns (success, new_domains, new_assignment, new_unassigned, new_count) or None if conflict."""
            if c not in domains[v]:
                return None
            
            new_assignment = assignment[:]
            new_unassigned = unassigned[:]
            new_domains = [d[:] for d in domains]
            
            new_assignment[v] = c
            new_unassigned[v] = False
            new_count = unassigned_count - 1
            new_domains[v] = [c]
            
            for u in adj[v]:
                if new_unassigned[u]:
                    if c in new_domains[u]:
                        new_domains[u].remove(c)
                        if len(new_domains[u]) == 0:
                            return None
            
            return (True, new_domains, new_assignment, new_unassigned, new_count)
        
        def backtrack(order_idx, domains, assignment, unassigned, unassigned_count):
            if time.monotonic() > deadline:
                return 'timeout'
            
            if unassigned_count == 0:
                return assignment[:]
            
            # Find next unassigned vertex in order
            v = -1
            for i in range(order_idx, n):
                if unassigned[order[i]]:
                    v = order[i]
                    order_idx = i
                    break
            
            if v == -1:
                return assignment[:]
            
            # MRV: among remaining unassigned, pick one with smallest domain
            best_v = v
            best_size = len(domains[v])
            for i in range(order_idx + 1, n):
                u = order[i]
                if unassigned[u]:
                    sz = len(domains[u])
                    if sz < best_size:
                        best_size = sz
                        best_v = u
                        order_idx = i
                    elif sz == best_size and degree[u] > degree[best_v]:
                        best_v = u
                        order_idx = i
            
            if best_size == 0:
                return False
            
            # Try colors in domain
            for c in domains[best_v]:
                result = try_color(best_v, c, domains, assignment, unassigned, unassigned_count)
                if result is None:
                    continue
                
                success, new_domains, new_assignment, new_unassigned, new_count = result
                if not success:
                    continue
                
                res = backtrack(order_idx, new_domains, new_assignment, new_unassigned, new_count)
                if res == 'timeout':
                    return 'timeout'
                if res is not None:
                    return res
            
            return False
        
        # Shuffle domain order for diversity across restarts
        for i in range(n):
            rng.shuffle(domains[i])
        
        result = backtrack(0, domains, assignment, unassigned, unassigned_count)
        if result == 'timeout':
            return None
        if result is not None:
            return result
        return None

    out = []
    start_time = time.monotonic()
    hard_deadline = start_time + 130.0  # 20s margin before 150s kill

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        if time.monotonic() > hard_deadline - 5.0:
            out.append([0] * n)
            continue
        
        remaining_instances = 4 - len(out)
        time_left = hard_deadline - time.monotonic()
        instance_deadline = time.monotonic() + (time_left / remaining_instances) * 0.85
        
        colouring = None
        for retry_seed in range(20):
            if time.monotonic() > instance_deadline:
                break
            rng_solver = Random(retry_seed * 7 + 13)
            colouring = solve_instance(n, edges, lists, instance_deadline, rng_solver)
            if colouring is not None:
                break
        
        if colouring is None:
            # Greedy fallback
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colouring = [-1] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colouring[u] for u in adj[v] if colouring[u] != -1}
                free = [c for c in lists[v] if c not in taken]
                colouring[v] = free[0] if free else lists[v][0]
        
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
