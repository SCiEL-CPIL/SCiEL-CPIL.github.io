import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    random.seed(42)
    
    for target_n in [18, 17]:
        if time.time() > deadline - 5:
            break
        result = _greedy(target_n, deadline)
        if result is not None:
            return result
    
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _greedy(n, deadline):
    total_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        perm = list(range(total_edges))
        random.shuffle(perm)
        
        colors = [-1] * total_edges
        adj = [[set() for _ in range(n)] for _ in range(4)]
        
        ok = True
        for idx in perm:
            u, v = edges[idx]
            valid = []
            for c in range(4):
                if not _c6(u, v, adj[c]):
                    valid.append(c)
            if not valid:
                ok = False
                break
            c = valid[0]
            colors[idx] = c
            adj[c][u].add(v)
            adj[c][v].add(u)
        
        if ok:
            return {'n': n, 'colors': colors}
    return None

def _c6(u, v, adj_c):
    for a in adj_c[u]:
        if a == v: continue
        for b in adj_c[a]:
            if b in (u, v): continue
            for d in adj_c[b]:
                if d in (u, a, v): continue
                for e in adj_c[d]:
                    if e in (u, a, b, v): continue
                    if v in adj_c[e]:
                        return True
    return False
