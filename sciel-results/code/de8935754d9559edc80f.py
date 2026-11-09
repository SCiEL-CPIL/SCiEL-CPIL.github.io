"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adjacency(n, base_edges, added_edges):
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_distances(n, adj, src):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _compute_all_distances(n, adj):
    dists = [[0]*n for _ in range(n)]
    for i in range(n):
        dists[i] = _bfs_distances(n, adj, i)
    return dists

def _get_violating_pairs(n, dists, limit=4):
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if dists[i][j] > limit:
                pairs.append((i, j))
    return pairs

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for e in base_edges:
        base_set.add(tuple(sorted(e)))
    
    # Generate candidates: even-odd, not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_set:
                candidates.append((u, v))
                
    # 1. Greedy Add
    adj = _build_adjacency(n, base_edges, [])
    dists = _compute_all_distances(n, adj)
    violating = _get_violating_pairs(n, dists, 4)
    
    added = []
    added_set = set()
    
    while violating:
        best_edge = None
        best_fixes = -1
        best_dists = None
        
        for e in candidates:
            if e in added_set:
                continue
            
            # Test adding e
            test_added = added + [e]
            test_adj = _build_adjacency(n, base_edges, test_added)
            test_dists = _compute_all_distances(n, test_adj)
            
            # Count fixes
            fixes = 0
            still_violating = []
            for p in violating:
                if test_dists[p[0]][p[1]] <= 4:
                    fixes += 1
                else:
                    still_violating.append(p)
            
            if fixes > best_fixes:
                best_fixes = fixes
                best_edge = e
                best_dists = test_dists
                best_violating = still_violating

        if best_edge is None:
            # Should not happen if graph is connected and bipartite candidates suffice
            break
            
        added.append(best_edge)
        added_set.add(best_edge)
        dists = best_dists
        violating = best_violating
        
    # 2. Local Search: Remove edges if possible
    improved = True
    while improved:
        improved = False
        for i in range(len(added)):
            test_added = added[:i] + added[i+1:]
            test_adj = _build_adjacency(n, base_edges, test_added)
            test_dists = _compute_all_distances(n, test_adj)
            
            # Check diameter
            max_d = 0
            for r in range(n):
                for c in range(n):
                    if test_dists[r][c] > max_d:
                        max_d = test_dists[r][c]
            
            if max_d <= 4:
                added = test_added
                improved = True
                break
                
    return [[u, v] for u, v in added]
# EVOLVE-BLOCK-END
