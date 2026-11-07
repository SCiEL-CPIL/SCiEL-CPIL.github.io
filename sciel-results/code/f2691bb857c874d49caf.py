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

    def solve(n, edges, lists):
        import sys
        sys.setrecursionlimit(10000)
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # avail[v] is a set of available colors
        avail = [set(lists[v]) for v in range(n)]
        color = [None] * n
        unassigned = set(range(n))
        
        def backtrack():
            if not unassigned:
                return True
            
            # MRV: Pick unassigned vertex with smallest available domain
            v = None
            min_size = float('inf')
            for u in unassigned:
                size = len(avail[u])
                if size < min_size:
                    min_size = size
                    v = u
                    if size == 0:
                        return False
            
            # Try each color in the domain of v
            for c in sorted(avail[v]):
                color[v] = c
                unassigned.remove(v)
                
                # Forward Checking: remove c from neighbors
                removed = []
                failed = False
                for u in adj[v]:
                    if u in unassigned and c in avail[u]:
                        avail[u].remove(c)
                        removed.append((u, c))
                        if not avail[u]:
                            failed = True
                            break
                
                if not failed:
                    if backtrack():
                        return True
                
                # Undo
                color[v] = None
                unassigned.add(v)
                for u, c in removed:
                    avail[u].add(c)
            
            return False
        
        if backtrack():
            return color
        else:
            # Fallback: should not happen
            return [0] * n

    out = []
    start_time = time.time()
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() - start_time > 110:
            # Safety break
            break
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists)
        # Ensure types are correct: list of ints
        out.append([int(c) for c in sol])
        
    return out
