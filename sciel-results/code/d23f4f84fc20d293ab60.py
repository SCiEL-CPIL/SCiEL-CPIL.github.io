"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
import collections

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

    def solve_one(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [list(lst) for lst in lists]
        assigned = [None] * n
        
        def propagate():
            # Return False if inconsistency found
            queue = collections.deque()
            for v in range(n):
                if assigned[v] is None and len(domains[v]) == 1:
                    queue.append(v)
            
            while queue:
                v = queue.popleft()
                if assigned[v] is not None:
                    continue
                if len(domains[v]) == 0:
                    return False
                if len(domains[v]) == 1:
                    c = domains[v][0]
                    assigned[v] = c
                    for u in adj[v]:
                        if assigned[u] is None:
                            if c in domains[u]:
                                domains[u].remove(c)
                                if len(domains[u]) == 0:
                                    return False
                                if len(domains[u]) == 1:
                                    queue.append(u)
            return True
        
        # Initial propagation
        if not propagate():
            return None
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find unassigned vertex with minimum domain size (MRV)
            min_v = -1
            min_size = float('inf')
            for v in range(n):
                if assigned[v] is None:
                    sz = len(domains[v])
                    if sz < min_size:
                        min_size = sz
                        min_v = v
                        if sz <= 1:
                            break
            
            if min_v == -1:
                return True  # All assigned
            
            if min_size == 0:
                return False  # No available colour
            
            # Try each colour in the domain
            for c in list(domains[min_v]):
                assigned[min_v] = c
                
                # Forward checking: remove c from neighbors' domains
                saved = {}
                consistent = True
                for u in adj[min_v]:
                    if assigned[u] is None:
                        if c in domains[u]:
                            saved[u] = domains[u][:]
                            domains[u].remove(c)
                            if len(domains[u]) == 0:
                                consistent = False
                                break
                
                if consistent:
                    # Save state before propagation for undo
                    saved_assigned = [None] * n
                    saved_domains = [None] * n
                    for v in range(n):
                        if assigned[v] is not None:
                            saved_assigned[v] = assigned[v]
                        if domains[v] is not None:
                            saved_domains[v] = domains[v][:]
                    
                    if propagate():
                        if backtrack():
                            return True
                        # Undo propagation
                        for v in range(n):
                            assigned[v] = saved_assigned[v]
                            if saved_domains[v] is not None:
                                domains[v] = saved_domains[v]
                    
                    # Undo forward checking
                    for u in saved:
                        domains[u] = saved[u]
                
                # Undo forward checking
                for u in saved:
                    domains[u] = saved[u]
                assigned[min_v] = None
            
            return False
        
        if backtrack():
            # Verify solution
            for v in range(n):
                if assigned[v] is None:
                    return None
                if assigned[v] not in lists[v]:
                    return None
                for u in adj[v]:
                    if u > v and assigned[u] == assigned[v]:
                        return None
            return assigned
        
        return None

    out = []
    for idx, (splits, seed) in enumerate(((20, 701), (40, 702), (60, 703), (100, 704))):
        if time.time() > deadline - 5:
            break
        
        n, edges, lists = instance(splits, seed)
        result = solve_one(n, edges, lists)
        
        if result is None:
            # Fallback: just assign first colour from list (may be invalid)
            result = [lst[0] for lst in lists]
        
        # Ensure all values are Python int
        out.append([int(c) for c in result])
    
    # If we ran out of time, fill remaining with simple assignments
    while len(out) < 4:
        idx = len(out)
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[idx]
        n, edges, lists = instance(splits, seed)
        out.append([int(lst[0]) for lst in lists])
    
    return out
# EVOLVE-BLOCK-END
