"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check_valid(n, base_edges, added_edges, max_diam=4):
    """Fast validity check: no duplicates, connected, triangle-free, diameter <= max_diam."""
    all_edges = base_edges + added_edges
    edge_set = set()
    for u, v in all_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Connectedness
    visited = [False] * n
    visited[0] = True
    q = deque([0])
    cnt = 1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                cnt += 1
                q.append(v)
    if cnt != n:
        return False
    
    # Triangle-free
    for u, v in all_edges:
        if adj[u] & adj[v]:
            return False
    
    # Diameter
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_diam:
                        return False
                    q.append(v)
        if -1 in dist:
            return False
    return True

def _local_search(n, base_edges, added_edges, max_diam=4, deadline=None):
    """Aggressive local search: try removing each edge, then 1-for-1 swaps."""
    current = [list(e) for e in added_edges]
    
    # Repeatedly try to remove edges
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current)):
            if deadline and time.time() > deadline:
                break
            test = current[:i] + current[i+1:]
            if _check_valid(n, base_edges, test, max_diam):
                current = test
                improved = True
                break
    
    # Try 1-for-1 swaps to enable further removals
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current)):
            if deadline and time.time() > deadline:
                break
            without_i = current[:i] + current[i+1:]
            all_no_i = list(base_edges) + without_i
            adj_no_i = [set() for _ in range(n)]
            for a, b in all_no_i:
                adj_no_i[a].add(b)
                adj_no_i[b].add(a)
            edge_set_no_i = set((min(a,b), max(a,b)) for a, b in all_no_i)
            
            found = False
            for u in range(n):
                if deadline and time.time() > deadline or found:
                    break
                for v in range(u+1, n):
                    if deadline and time.time() > deadline or found:
                        break
                    key = (min(u,v), max(u,v))
                    if key in edge_set_no_i:
                        continue
                    if adj_no_i[u] & adj_no_i[v]:
                        continue
                    test = without_i + [[u, v]]
                    if _check_valid(n, base_edges, test, max_diam):
                        current = test
                        improved = True
                        found = True
                        break
    
    # Final removal pass
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current)):
            if deadline and time.time() > deadline:
                break
            test = current[:i] + current[i+1:]
            if _check_valid(n, base_edges, test, max_diam):
                current = test
                improved = True
                break
    
    return current

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    n = P['n']
    base_edges = P['edges']
    max_diam = P['diameter']
    
    # Strategy 1: Single hub at 12 (even), spokes to odd vertices at distance >= 4
    hub = 12
    spokes1 = []
    for v in range(1, 24, 2):  # Odd vertices
        if v != 11 and v != 13 and abs(v - hub) >= 4:
            spokes1.append([hub, v])
    
    # Strategy 2: Try all even hubs 6..17
    best_result = None
    best_count = len(spokes1)
    
    if _check_valid(n, base_edges, spokes1, max_diam):
        best_result = spokes1
    
    for hub in range(6, 18):
        if hub % 2 != 0:
            continue
        if time.time() > deadline:
            break
        spokes = []
        for v in range(1, 24, 2):
            if v != hub - 1 and v != hub + 1 and abs(v - hub) >= 4:
                spokes.append([hub, v])
        if len(spokes) >= best_count:
            continue
        if _check_valid(n, base_edges, spokes, max_diam):
            best_count = len(spokes)
            best_result = spokes
    
    # Strategy 3: Two-hub configurations (even hub + odd hub, possibly with connecting edge)
    for h1 in range(8, 16):
        if h1 % 2 != 0:
            continue
        for h2 in range(h1 - 5, h1 + 6):
            if h2 <= 0 or h2 >= 24 or h2 % 2 == 0 or h2 == h1:
                continue
            if time.time() > deadline:
                break
            edges_h = []
            if abs(h1 - h2) > 1:
                edges_h.append([h1, h2])
            # Even vertices needing spoke (to h2, which is odd)
            for v in range(0, 24, 2):
                if v == h1:
                    continue
                if abs(v - h1) > 3 and abs(v - h2) > 3:
                    edges_h.append([h2, v])
            # Odd vertices needing spoke (to h1, which is even)
            for v in range(1, 24, 2):
                if v == h2:
                    continue
                if abs(v - h1) > 3 and abs(v - h2) > 3:
                    edges_h.append([h1, v])
            if len(edges_h) >= best_count:
                continue
            if _check_valid(n, base_edges, edges_h, max_diam):
                best_count = len(edges_h)
                best_result = edges_h
    
    # Local search on best result
    if best_result is not None and time.time() < deadline:
        best_result = _local_search(n, base_edges, best_result, max_diam, deadline)
        if _check_valid(n, base_edges, best_result, max_diam):
            best_count = len(best_result)
    
    # Fallback: guaranteed valid 10-edge solution
    if best_result is None:
        best_result = [[12, v] for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
    
    return [[int(u), int(v)] for u, v in best_result]
# EVOLVE-BLOCK-END
