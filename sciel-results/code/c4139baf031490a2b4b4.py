"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n):
    """Compute all-pairs shortest paths via BFS from each vertex."""
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    queue.append(v)
    return dist

def _check_solution(added_edges, n):
    """Check triangle-free, connected, diameter <= 4."""
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check triangle-free
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for w in adj[u][i+1:]:
                if w in adj[v]:
                    return False
    
    # Check connected and diameter
    dist = _bfs_distances(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1 or dist[i][j] > 4:
                return False
    return True

def _build_adj(added_edges, n):
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _pairs_exceeding(adj, n, limit=4):
    """Return set of pairs with distance > limit."""
    dist = _bfs_distances(adj, n)
    bad = set()
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] > limit:
                bad.add((i, j))
    return bad

def _edge_fixes(e, adj, n, limit=4):
    """Which pairs does adding edge e fix (distance becomes <= limit)?"""
    u, v = e
    # Pairs (a,b) where shortest path goes through edge (u,v)
    # New distance: min(old, dist(a,u)+1+dist(v,b), dist(a,v)+1+dist(u,b))
    dist = _bfs_distances(adj, n)
    fixed = set()
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] <= limit:
                continue
            # Try using edge (u,v)
            d1 = dist[i][u] + 1 + dist[v][j] if dist[i][u] >= 0 and dist[v][j] >= 0 else 999
            d2 = dist[i][v] + 1 + dist[u][j] if dist[i][v] >= 0 and dist[u][j] >= 0 else 999
            if min(d1, d2) <= limit:
                fixed.add((i, j))
    return fixed

def construct():
    start_time = time.time()
    n = P['n']
    
    # All candidate edges: even-odd pairs not already in path
    path_edges = set()
    for u, v in P['edges']:
        path_edges.add((min(u,v), max(u,v)))
    
    candidates = []
    for e in range(0, n, 2):
        for o in range(1, n, 2):
            pair = (min(e, o), max(e, o))
            if pair not in path_edges:
                candidates.append(pair)
    
    # Greedy set cover: repeatedly pick edge that fixes most uncovered pairs
    adj = _build_adj([], n)
    bad = _pairs_exceeding(adj, n)
    
    if not bad:
        return []
    
    selected = []
    remaining_bad = set(bad)
    
    while remaining_bad and (time.time() - start_time) < 80:
        best_edge = None
        best_fixed = set()
        for e in candidates:
            if e in selected:
                continue
            # Check bipartite: even-odd already guaranteed
            fixed = _edge_fixes(e, adj, n)
            new_fixed = fixed & remaining_bad
            if len(new_fixed) > len(best_fixed):
                best_fixed = new_fixed
                best_edge = e
                if len(best_fixed) == len(remaining_bad):
                    break
        if best_edge is None or not best_fixed:
            break
        selected.append(best_edge)
        adj = _build_adj(selected, n)
        remaining_bad -= best_fixed
    
    # Verify
    if _check_solution(selected, n):
        result = [[int(u), int(v)] for u, v in selected]
        return result
    
    # Fallback: hub at 12
    hub = 12
    fallback = []
    for v in range(1, 24, 2):
        if v != 11 and v != 13:
            fallback.append([hub, v])
    return [[int(u), int(v)] for u, v in fallback]
# EVOLVE-BLOCK-END
