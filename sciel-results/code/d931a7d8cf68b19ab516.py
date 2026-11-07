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

    def solve_csp(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        colour = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # MRV heuristic: pick uncolored vertex with fewest available colors
            best_v = -1
            best_avail = None
            for v in range(idx, n):
                if colour[v] != -1:
                    continue
                taken = {colour[u] for u in adj[v] if colour[u] != -1}
                avail = [c for c in lists[v] if c not in taken]
                if len(avail) == 0:
                    return False
                if best_v == -1 or len(avail) < len(best_avail):
                    best_v = v
                    best_avail = avail
                    if len(avail) == 1:
                        break
            
            if best_v == -1:
                # All remaining are colored
                return True
            
            # Try each available color for best_v
            for c in best_avail:
                colour[best_v] = c
                if backtrack(idx + 1):
                    return True
                colour[best_v] = -1
            
            return False
        
        return backtrack(0)

    out = []
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we're running out of time, use greedy for remaining
            n, edges, lists = instance(splits, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [-1] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] != -1}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
            continue
        
        n, edges, lists = instance(splits, seed)
        if solve_csp(n, edges, lists):
            out.append(colour)
        else:
            # Fallback to greedy
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [-1] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] != -1}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
    
    return out
# EVOLVE-BLOCK-END
