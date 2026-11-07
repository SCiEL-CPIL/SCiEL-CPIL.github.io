"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import sys
    from random import Random

    sys.setrecursionlimit(10000)

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

    def solve(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        color = [-1] * n
        uncolored = set(range(n))
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV Heuristic: Pick uncolored vertex with fewest available colors
            best_v = -1
            min_options = 4
            best_options = None
            
            for v in uncolored:
                taken = {color[u] for u in adj[v] if color[u] != -1}
                options = [c for c in lists[v] if c not in taken]
                if not options:
                    return False
                if len(options) < min_options:
                    min_options = len(options)
                    best_v = v
                    best_options = options
                    if min_options == 1:
                        break
            
            if best_v == -1:
                return False # Should not happen if uncolored is not empty
            
            uncolored.remove(best_v)
            
            for c in best_options:
                color[best_v] = c
                if backtrack():
                    return True
                color[best_v] = -1
            
            uncolored.add(best_v)
            return False

        if backtrack():
            return [int(c) for c in color]
        else:
            return [0] * n

    out = []
    start_time = time.time()
    deadline = start_time + 110.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback: generate valid structure but potentially invalid coloring
            # to avoid timeout crash, though we expect fast completion.
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        res = solve(n, edges, lists)
        out.append(res)
        
    return out
# EVOLVE-BLOCK-END
