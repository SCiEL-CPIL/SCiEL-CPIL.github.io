import time
import sys
from random import Random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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
        lists = [set(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Domains as lists of available colors
        domains = [list(lst) for lst in lists]
        colors = [-1] * n
        
        # AC-3 for initial arc consistency
        def ac3(domains, adj):
            queue = deque()
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                if not domains[u]:
                    return False
                removed = False
                new_domains_u = []
                for c in domains[u]:
                    # Check if c has a support in domains[v] (i.e., exists d in domains[v] with d != c)
                    # Since it's a != constraint, support exists if domains[v] has any element != c
                    # which is true if len(domains[v]) > 1 or (len(domains[v]) == 1 and domains[v][0] != c)
                    if len(domains[v]) > 1 or (len(domains[v]) == 1 and domains[v][0] != c):
                        new_domains_u.append(c)
                    else:
                        removed = True
                if removed:
                    domains[u] = new_domains_u
                    if not domains[u]:
                        return False
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        # Backtracking with MRV, Degree, and Forward Checking
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find unassigned vertex with MRV (then degree)
            best_v = -1
            min_size = n + 1
            best_degree = -1
            
            for v in range(n):
                if colors[v] == -1:
                    size = len(domains[v])
                    if size == 0:
                        return False
                    deg = len(adj[v])
                    if size < min_size or (size == min_size and deg > best_degree):
                        min_size = size
                        best_degree = deg
                        best_v = v
                        if size == 1:
                            break
            
            if best_v == -1:
                return True
            
            # Order colors by Least Constraining Value (LCV) heuristic
            # Estimate how many options each color leaves for neighbors
            def lcv_sort(c_list, v):
                scored = []
                for c in c_list:
                    count = 0
                    for u in adj[v]:
                        if colors[u] == -1:
                            if c in domains[u]:
                                count += 1
                    # We want to minimize count (least constraining)
                    scored.append((-count, c))
                scored.sort()
                return [c for _, c in scored]
            
            sorted_colors = lcv_sort(domains[best_v], best_v)
            
            saved_domain = domains[best_v][:]
            for c in sorted_colors:
                colors[best_v] = c
                domains[best_v] = [c]
                
                # Forward Checking
                valid = True
                backup = []
                for u in adj[best_v]:
                    if colors[u] == -1 and c in domains[u]:
                        old = domains[u][:]
                        domains[u] = [d for d in domains[u] if d != c]
                        backup.append((u, old))
                        if not domains[u]:
                            valid = False
                            break
                
                if valid:
                    if backtrack():
                        return True
                
                # Undo
                colors[best_v] = -1
                domains[best_v] = saved_domain
                for u, old in backup:
                    domains[u] = old
            
            return False

        # First, try to prune domains with AC-3
        if not ac3(domains, adj):
            # If AC-3 fails, it's unsolvable (shouldn't happen per problem statement)
            # But just in case, return dummy
            return [0] * n

        if backtrack():
            return [int(c) for c in colors]
        else:
            return [0] * n

    out = []
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue

        n, edges, lists = instance(splits, seed)
        
        if time.time() > deadline - 5.0:
            # Simple greedy fallback
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colors = [-1] * n
            for v in range(n):
                taken = {colors[u] for u in adj[v] if colors[u] != -1}
                for c in sorted(lists[v]):
                    if c not in taken:
                        colors[v] = c
                        break
            out.append([int(c) for c in colors])
            continue

        res = solve(n, edges, lists)
        
        # Ensure types are strictly Python int
        res = [int(x) for x in res]
        
        if len(res) != n:
            res = [0] * n
            
        out.append(res)

    return out
