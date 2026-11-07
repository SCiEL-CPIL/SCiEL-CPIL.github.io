"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 85.0

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj_masks, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        mask = adj_masks[u]
        m = mask
        while m:
            lb = m & -m
            v = lb.bit_length() - 1
            m ^= lb
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_solution_fast(n, base_edges, added_edges):
    adj_masks = [0] * n
    
    def add_edge(u, v):
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    for u, v in base_edges:
        add_edge(u, v)
        
    for u, v in added_edges:
        if (adj_masks[u] >> v) & 1:
            return False
        add_edge(u, v)
        
    for i in range(n):
        for j in range(i+1, n):
            if (adj_masks[i] >> j) & 1:
                if (adj_masks[i] & adj_masks[j]) != 0:
                    return False
                    
    for start in range(n):
        dist = _bfs_distances(adj_masks, n, start)
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
            
    return True

def _is_triangle_free_add_mask(adj_masks, u, v):
    return (adj_masks[u] & adj_masks[v]) == 0

def _compute_diameter(adj_masks, n):
    max_d = 0
    for start in range(n):
        dist = _bfs_distances(adj_masks, n, start)
        if -1 in dist:
            return -1
        d = max(dist)
        if d > max_d:
            max_d = d
    return max_d

def _greedy_bipartite(n, base):
    base_set = set()
    for u, v in base:
        base_set.add((min(u, v), max(u, v)))
        
    adj_masks = [0] * n
    for u, v in base:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    # Precompute all valid even-odd edges
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in base_set:
                continue
            if i % 2 != j % 2:  # even-odd pair
                if _is_triangle_free_add_mask(adj_masks, i, j):
                    candidates.append((i, j))
    
    added = []
    
    # Compute current distances
    def get_dist_matrix():
        dm = [[0]*n for _ in range(n)]
        for s in range(n):
            dist = _bfs_distances(adj_masks, n, s)
            for t in range(n):
                dm[s][t] = dist[t] if dist[t] != -1 else 100
        return dm
    
    dm = get_dist_matrix()
    
    while True:
        if not _time_ok():
            break
        if max(max(row) for row in dm) <= 4:
            break
        
        best_edge = None
        best_score = -1
        
        for i, j in candidates:
            if (i, j) in base_set:
                continue
            # Check if this edge is still valid
            if not _is_triangle_free_add_mask(adj_masks, i, j):
                continue
            
            # Estimate improvement: count pairs where dist > 4 and this edge helps
            # Quick heuristic: sum of (dm[i][k] + 1 + dm[j][k]) for k where dm[i][k] + dm[j][k] + 2 < dm[i][k] or similar
            # Simpler: just pick edge that maximizes min(dm[i][k], dm[j][k]) reduction potential
            # Use: for each k, new_dist = min(dm[i][k], 1 + dm[j][k], dm[j][k], 1 + dm[i][k])
            # Gain = sum over k of max(0, max(dm[i][k], dm[j][k]) - new_dist)
            
            gain = 0
            for k in range(n):
                old_d = max(dm[i][k], dm[j][k])
                new_d = min(dm[i][k], 1 + dm[j][k], dm[j][k], 1 + dm[i][k])
                if old_d > new_d:
                    gain += old_d - new_d
            
            if gain > best_score:
                best_score = gain
                best_edge = (i, j)
        
        if best_edge is None or best_score == 0:
            break
        
        u, v = best_edge
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        added.append([u, v])
        
        # Recompute distance matrix
        dm = get_dist_matrix()
    
    return added

def _prune_edges(n, base, added):
    if not added:
        return added
    
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(added) - 1, -1, -1):
            if not _time_ok():
                break
            edge_to_remove = added[i]
            new_added = added[:i] + added[i+1:]
            if _check_solution_fast(n, base, new_added):
                added = new_added
                changed = True
    return added

def construct():
    n = P['n']
    base = P['edges']
    
    # 1. Greedy bipartite construction
    added = _greedy_bipartite(n, base)
    
    # 2. Prune
    added = _prune_edges(n, base, added)
    
    # 3. Verify
    if _check_solution_fast(n, base, added):
        return [[int(u), int(v)] for u, v in added]
    
    # Fallback: dense bipartite + prune
    base_set = set()
    for u, v in base:
        base_set.add((min(u, v), max(u, v)))
    adj_masks = [0] * n
    for u, v in base:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    dense = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in base_set:
                continue
            if i % 2 == j % 2:
                continue
            adj_masks[i] |= (1 << j)
            adj_masks[j] |= (1 << i)
            dense.append([i, j])
    dense = _prune_edges(n, base, dense)
    if _check_solution_fast(n, base, dense):
        return [[int(u), int(v)] for u, v in dense]
    
    return []

# EVOLVE-BLOCK-END
