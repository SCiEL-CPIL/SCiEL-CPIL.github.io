import time
import random
import numpy as np
from collections import deque

def _build_luw(k, g):
    """Construct Lazebnik-Ustimenko-Woldar graph as fallback."""
    q = k
    if q == 4:
        d = 5 if g <= 10 else 6
        tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def _add(a,b): return a ^ b
        def _sub(a,b): return a ^ b
        def _mul(a,b): return tbl[a][b]
    else:
        d = max(3, g - 5 if (g-5) % 2 else g - 4)
        def _add(a,b): return (a+b) % q
        def _sub(a,b): return (a-b) % q
        def _mul(a,b): return (a*b) % q
    
    rules = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        rules[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def _nbrs(v):
        side, x = v
        for f in range(q):
            y = [None, f] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rules[c]
                if side == 0:
                    y[c] = _add(x[c], _mul(y[a], x[b]))
                else:
                    y[c] = _sub(x[c], _mul(x[a], y[b]))
            yield (1-side, tuple(y))
    
    start = (0, (None,) + (0,)*d)
    idx = {start: 0}
    ordered = [start]
    edge_list = []
    for v in ordered:
        for nb in _nbrs(v):
            if nb not in idx:
                idx[nb] = len(ordered)
                ordered.append(nb)
            if v[0] == 0:
                edge_list.append((idx[v], idx[nb]))
    return len(ordered), edge_list


def _verify_girth(adj, n, g):
    """Verify no cycle shorter than g using non-backtracking walks from every vertex."""
    # BFS from each vertex, tracking parent to avoid backtracking
    for s in range(n):
        # dist[u] = shortest non-backtracking distance from s to u
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    dq.append(v)
                elif dist[v] <= dist[u]:
                    # Found a cycle of length dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True


def _count_short_cycles(adj, n, max_len=9):
    """Count cycles of length <= max_len."""
    count = 0
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if dist[u] >= max_len:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    dq.append(v)
                elif dist[v] <= dist[u]:
                    clen = dist[u] + dist[v] + 1
                    if clen <= max_len and clen >= 3:
                        count += 1
    return count


def _find_short_cycle_edges(adj, n, max_len=9):
    """Find edges that lie on short cycles."""
    cycle_edges = set()
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if dist[u] >= max_len:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    dq.append(v)
                elif dist[v] <= dist[u]:
                    clen = dist[u] + dist[v] + 1
                    if clen <= max_len and clen >= 3:
                        cycle_edges.add((min(u,v), max(u,v)))
    return cycle_edges


def _gen_4reg(n, seed=None):
    """Generate a random simple 4-regular graph on n vertices using configuration model."""
    if n < 4 or n % 2 != 0:
        return None
    rng = random.Random(seed)
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    rng.shuffle(stubs)
    adj = [[] for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != 4:
            return None
    return adj


def _local_search(adj, n, g, deadline, max_iter=2000):
    """Local search to eliminate short cycles via 2-switches."""
    rng = random.Random()
    current_cycles = _count_short_cycles(adj, n, g-1)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if current_cycles == 0:
            return adj, True
        
        # Find a short cycle edge
        ce = _find_short_cycle_edges(adj, n, g-1)
        if not ce:
            return adj, True
        
        # Pick a random cycle edge to break
        u, v = rng.choice(list(ce))
        
        # Find neighbors of u and v (excluding each other)
        nbrs_u = [x for x in adj[u] if x != v]
        nbrs_v = [y for y in adj[v] if y != u]
        
        if not nbrs_u or not nbrs_v:
            continue
        
        best_swap = None
        best_cycle_count = current_cycles
        
        # Try several swaps
        for _ in range(20):
            x = rng.choice(nbrs_u)
            y = rng.choice(nbrs_v)
            
            # Check if swap is valid
            if x == y:
                continue
            if y in adj[u] or x in adj[v]:
                continue
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            new_cycles = _count_short_cycles(adj, n, g-1)
            
            if new_cycles < best_cycle_count:
                best_swap = (u, v, x, y)
                best_cycle_count = new_cycles
            
            # Revert swap
            adj[u].remove(x)
            adj[x].remove(u)
            adj[v].remove(y)
            adj[y].remove(v)
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
        
        if best_swap:
            u, v, x, y = best_swap
            # Perform the best swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            current_cycles = best_cycle_count
        else:
            # No improvement found, try a random perturbation
            if rng.random() < 0.1:
                # Random 2-switch
                a = rng.randint(0, n-1)
                b = rng.randint(0, n-1)
                if a == b or b not in adj[a]:
                    continue
                nbrs_a = [x for x in adj[a] if x != b]
                nbrs_b = [y for y in adj[b] if y != a]
                if not nbrs_a or not nbrs_b:
                    continue
                x = rng.choice(nbrs_a)
                y = rng.choice(nbrs_b)
                if x == y or y in adj[a] or x in adj[b]:
                    continue
                adj[a].remove(b)
                adj[b].remove(a)
                adj[x].remove(y)
                adj[y].remove(x)
                adj[a].append(x)
                adj[x].append(a)
                adj[b].append(y)
                adj[y].append(b)
                current_cycles = _count_short_cycles(adj, n, g-1)
    
    return adj, current_cycles == 0


def construct(k, g):
    """Construct k-regular graph with girth at least g, minimizing vertices."""
    
    if k == 4 and g == 10:
        t0 = time.time()
        deadline = t0 + 110  # Leave 10s buffer for 120s limit
        
        # Try to find a graph with girth 10 starting from small n
        # The smallest known is 320 (Foster cage)
        # Try random construction + local search for n < 320
        
        best_n = None
        best_edges = None
        
        # Try a few n values below 320
        for n in [244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318]:
            if time.time() > deadline - 5:
                break
            
            # Try a few random instances
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                
                adj = _gen_4reg(n, seed=random.randint(0, 10**9))
                if adj is None:
                    continue
                
                # Check if already has girth >= 10
                if _verify_girth(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    break
                
                # Try local search
                adj, success = _local_search(adj, n, g, deadline - 3, max_iter=500)
                if success:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    break
            
            if best_n is not None:
                break
        
        if best_n is not None:
            return best_n, best_edges
        
        # Fallback to LUW construction
        return _build_luw(k, g)
    else:
        return _build_luw(k, g)
