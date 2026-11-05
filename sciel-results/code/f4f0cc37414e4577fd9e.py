"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
sys.setrecursionlimit(10000)

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)

edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    edge_idx[(min(u, v), max(u, v))] = i + 1

def compute_score(cycle):
    return max(edge_idx[(min(cycle[i], cycle[(i+1)%N]), max(cycle[i], cycle[(i+1)%N]))] for i in range(N))

def find_hc(K, tlim, rng):
    t0 = time.time()
    adj = [set() for _ in range(N)]
    for i in range(K):
        u, v = EDGE_ORDER[i]
        adj[u].add(v); adj[v].add(u)
    for v in range(N):
        if len(adj[v]) < 2: return None
    al = [list(s) for s in adj]
    
    for start in rng.sample(range(N), N):
        if time.time() - t0 > tlim: break
        vis = [False]*N; vis[start]=True
        path = [start]
        
        def bt():
            if time.time()-t0 > tlim: return False
            if len(path)==N:
                return start in adj[path[-1]]
            cur = path[-1]; rem = N-len(path)
            cands = []
            for nb in al[cur]:
                if vis[nb]: continue
                if rem==1:
                    if start in adj[nb]: cands.append(nb)
                else:
                    uc = sum(1 for x in al[nb] if not vis[x] and x!=cur)
                    if uc >= rem-1: cands.append(nb)
            cands.sort(key=lambda x: sum(1 for y in al[x] if not vis[y] and y!=cur))
            for nb in cands:
                path.append(nb); vis[nb]=True
                if bt(): return True
                path.pop(); vis[nb]=False
            return False
        
        if bt(): return path[:]
    return None

def construct():
    dl = time.time()+85
    rng = random.Random(42)
    best_c, best_s = None, M+1
    
    if time.time()<dl-5:
        c = find_hc(M, min(10, dl-5-time.time()), rng)
        if c: best_c, best_s = c, compute_score(c)
    
    lo, hi = N, M
    while lo<hi and time.time()<dl-10:
        mid=(lo+hi)//2
        rt = dl-10-time.time()
        if rt<0.5: break
        c = find_hc(mid, min(6, rt), rng)
        if c:
            s = compute_score(c)
            if s<best_s: best_c, best_s = c, s
            hi=mid
        else:
            lo=mid+1
    
    return best_c if best_c else list(range(N))
# EVOLVE-BLOCK-END
