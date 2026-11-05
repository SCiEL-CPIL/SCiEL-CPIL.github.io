"""Find a Hamilton cycle minimizing the max edge arrival index."""
import time
import random

def _init():
    n = 24
    rng = random.Random(614)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    rng.shuffle(edges)
    idx = {}
    for k, (u, v) in enumerate(edges):
        idx[(u, v)] = k + 1
        idx[(v, u)] = k + 1
    return n, edges, idx

_N, _EDGES, _IDX = _init()

def _try_find_cycle(T, deadline):
    """Try to find a Hamilton cycle in G_T (edges with index <= T)."""
    n = _N
    adj = [[] for _ in range(n)]
    for (u, v) in _EDGES:
        if _IDX[(u, v)] <= T:
            adj[u].append(v)
            adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) < 2:
            return None
    
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    def dfs(depth):
        if time.time() > deadline:
            return False
        if depth == n - 1:
            return 0 in adj[path[-1]]
        
        cur = path[-1]
        candidates = [v for v in adj[cur] if not visited[v]]
        if not candidates:
            return False
        
        remaining = [v for v in range(n) if not visited[v]]
        if len(remaining) > 1:
            for v in remaining:
                ok = False
                for w in adj[v]:
                    if not visited[w] or w == cur:
                        ok = True
                        break
                if not ok:
                    return False
        
        candidates.sort(key=lambda v: sum(1 for w in adj[v] if not visited[w]))
        
        for v in candidates:
            visited[v] = True
            path.append(v)
            if dfs(depth + 1):
                return True
            path.pop()
            visited[v] = False
        return False
    
    if dfs(1):
        return list(path)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    lo, hi = 24, 277
    best_cycle = None
    
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = _try_find_cycle(mid, deadline)
        if cycle is not None:
            hi = mid
            best_cycle = cycle
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    for T in [120, 150, 180, 200, 220, 240, 260, 276]:
        if time.time() > deadline:
            break
        cycle = _try_find_cycle(T, deadline)
        if cycle is not None:
            return cycle
    
    return list(range(_N))
