"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _check(n, base, added):
    adj = [set() for _ in range(n)]
    for u, v in base:
        adj[u].add(v); adj[v].add(u)
    for u, v in added:
        adj[u].add(v); adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    for s in range(n):
        dist = [-1]*n
        dist[s] = 0
        q = [s]; qi = 0
        while qi < len(q):
            u = q[qi]; qi += 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u]+1
                    q.append(v)
        for d in dist:
            if d < 0 or d > 4:
                return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 85
    n = P['n']
    base = P['edges']
    base_set = {tuple(e) for e in base}
    cands = [(u,v) for u in range(n) for v in range(u+1,n) 
             if u%2==v%2 and v-u>=3 and (u,v) not in base_set]
    
    random.seed(42)
    
    for k in [5, 6, 7, 8, 9, 10]:
        if time.time() > deadline:
            break
        for _ in range(20000):
            if time.time() > deadline:
                break
            sel = random.sample(cands, k)
            if _check(n, base, sel):
                return [[u,v] for u,v in sel]
    
    return [[u,v] for u in range(n) for v in range(u+1,n) if (u-v)%2 and (u,v) not in base_set]
# EVOLVE-BLOCK-END
