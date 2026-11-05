import time, math, random
from collections import deque
import numpy as np

def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar: guaranteed valid fallback.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y:(x+y)%q
        sub = lambda x,y:(x-y)%q
        mul = lambda x,y: x*y%q
    rule = {2:(1,1),3:(2,1),4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,)+(0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0]==0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _girth_ok_bitset(adj, n, g):
    # Fast girth certification using numpy bitsets for non-backtracking BFS
    max_depth = (g-1)//2
    if max_depth < 1:
        return True
    # Build adjacency as numpy arrays
    adj_np = [np.array(v, dtype=np.int32) for v in adj]
    
    for s in range(n):
        # Non-backtracking BFS from s
        # State: (current_node, prev_node) -> distance
        # Use bitsets to track visited states
        # dist[v] = minimum distance from s to v in non-backtracking sense
        dist = [-1]*n
        dist[s] = 0
        # Queue of (node, prev) pairs
        dq = deque()
        dq.append((s, -1))
        
        while dq:
            u, par = dq.popleft()
            du = dist[u]
            if du >= max_depth:
                continue
            for v in adj[u]:
                if v == par:
                    continue
                dv = dist[v]
                if dv == -1:
                    dist[v] = du + 1
                    dq.append((v, u))
                else:
                    # Found a cycle of length du + dv + 1
                    if du + dv + 1 < g:
                        return False
    return True

def _girth_ok_fast(adj, n, g):
    # Even faster: check all pairs within distance g-2
    # If any two vertices at distance < g share a common neighbor path, cycle exists
    # Use BFS from each vertex up to depth (g-1)//2
    max_depth = (g-1)//2
    if max_depth < 1:
        return True
    
    for s in range(n):
        dist = [-1]*n
        par = [-1]*n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du >= max_depth:
                continue
            for v in adj[u]:
                if v == par[u]:
                    continue
                dv = dist[v]
                if dv == -1:
                    dist[v] = du+1
                    par[v] = u
                    dq.append(v)
                elif du+dv+1 < g:
                    return False
    return True

def _short_path_bitset(adj_bitsets, n, u, v, maxd):
    # Check if dist(u,v) <= maxd using bitset BFS
    if u == v:
        return maxd >= 0
    if maxd < 1:
        return False
    
    # Bitset BFS
    visited = np.zeros(n, dtype=bool)
    visited[u] = True
    current = np.zeros(n, dtype=bool)
    current[u] = True
    
    for _ in range(maxd):
        # Expand: neighbors of current that aren't visited
        next_set = np.zeros(n, dtype=bool)
        for i in range(n):
            if current[i]:
                next_set[adj_bitsets[i]] = True
        next_set[visited] = False
        if next_set[v]:
            return True
        if not next_set.any():
            return False
        visited |= next_set
        current = next_set
    
    return False

def _build_adj_bitsets(adj, n):
    """Precompute numpy arrays for each vertex's adjacency for fast bitset operations"""
    return [np.array(v, dtype=np.int32) for v in adj]

def _greedy(n, k, g, rng, deadline):
    # Add only edges that cannot create a cycle shorter than g.
    adj = [[] for _ in range(n)]
    eset = set()
    deg = [0]*n
    target = n*k//2
    added = 0
    maxd = g-2
    
    # Precompute bitsets for short path checks
    adj_bitsets = _build_adj_bitsets(adj, n)
    
    while added < target:
        if time.time() > deadline:
            return None
        progress = False
        # Try multiple random pairs
        for _ in range(200):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v or deg[u] >= k or deg[v] >= k:
                continue
            key = (u, v) if u < v else (v, u)
            if key in eset:
                continue
            # Check if adding edge would create short cycle
            # This is equivalent to checking if dist(u,v) <= g-2 in current graph
            if _short_path_bitset(adj_bitsets, n, u, v, maxd):
                continue
            
            # Add the edge
            adj[u].append(v)
            adj[v].append(u)
            eset.add(key)
            deg[u] += 1
            deg[v] += 1
            added += 1
            progress = True
            # Update bitsets
            adj_bitsets[u] = np.append(adj_bitsets[u], v)
            adj_bitsets[v] = np.append(adj_bitsets[v], u)
            break
        
        if not progress:
            # Try with different random seeds or break
            return None
    
    return adj

def _random_k_regular(n, k, rng, tries=100):
    if (n*k) % 2:
        return None
    stubs = []
    for v in range(n):
        stubs.extend([v]*k)
    for _ in range(tries):
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        ok = True
        for i in range(0, n*k, 2):
            a, b = stubs[i], stubs[i+1]
            if a == b or b in adj[a]:
                ok = False
                break
            adj[a].append(b)
            adj[b].append(a)
        if ok:
            return adj
    return None

def _edges(adj, n):
    e = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                e.append((i, j))
    return e

def construct(k, g):
    start = time.time()
    deadline = start + 115.0  # Leave margin for verification
    
    # Calculate Moore bound
    if g % 2 == 1:
        r = (g-3)//2
        moore = k * sum((k-1)**i for i in range(r+1))
    else:
        r = g//2 - 1
        moore = k*(k-1) * sum((k-1)**i for i in range(r))
    
    n0 = int(math.ceil(moore))
    if n0 < 2:
        n0 = 2
    if (n0*k) % 2:
        n0 += 1
    
    max_vertices = 1200
    rng = random.Random(12345)
    
    # Try greedy construction first for each n
    for n in range(n0, max_vertices+1, 2):
        if time.time() > deadline - 5.0:
            break
        
        # Try greedy with multiple attempts
        for attempt in range(5):
            if time.time() > deadline - 3.0:
                break
            adj = _greedy(n, k, g, rng, deadline - 3.0)
            if adj is not None:
                # Verify girth
                if _girth_ok_fast(adj, n, g):
                    return n, _edges(adj, n)
        
        # Try random k-regular graphs
        for attempt in range(30):
            if time.time() > deadline - 2.0:
                break
            adj = _random_k_regular(n, k, rng, tries=50)
            if adj is not None:
                if _girth_ok_fast(adj, n, g):
                    return n, _edges(adj, n)
    
    # Fallback to LUW construction
    return _luw_construct(k, g)
