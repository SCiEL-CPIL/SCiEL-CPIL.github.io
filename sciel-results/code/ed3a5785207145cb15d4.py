"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 80.0

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj_masks(n, base_edges):
    adj_masks = [0] * n
    base_set = set()
    for u, v in base_edges:
        a, b = (u, v) if u < v else (v, u)
        base_set.add((a, b))
        adj_masks[a] |= (1 << b)
        adj_masks[b] |= (1 << a)
    return adj_masks, base_set

def _bfs_distances(adj_masks, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        m = adj_masks[u]
        while m:
            lb = m & -m
            v = lb.bit_length() - 1
            m ^= lb
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _compute_all_distances(adj_masks, n):
    all_dist = []
    for s in range(n):
        all_dist.append(_bfs_distances(adj_masks, n, s))
    return all_dist

def _get_bad_pairs(all_dist, n):
    bad = set()
    for u in range(n):
        for v in range(u + 1, n):
            if all_dist[u][v] > 4:
                bad.add((u, v))
    return bad

def _is_valid(n, base_edges, added_edges):
    adj_masks = [0] * n
    edge_set = set()
    for u, v in base_edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
        adj_masks[a] |= (1 << b)
        adj_masks[b] |= (1 << a)
    for u, v in added_edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
        adj_masks[a] |= (1 << b)
        adj_masks[b] |= (1 << a)
    # Triangle-free check
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_masks[i] >> j) & 1:
                if (adj_masks[i] & adj_masks[j]) != 0:
                    return False
    # Connectivity and diameter
    all_dist = _compute_all_distances(adj_masks, n)
    for dist in all_dist:
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
    return True

def _greedy_construct(n, base_edges):
    adj_masks, base_set = _build_adj_masks(n, base_edges)
    
    # Candidate edges: only even-odd (bipartite) non-base edges
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in base_set:
                continue
            if i % 2 == j % 2:
                continue
            candidates.append((i, j))
    
    added = []
    added_set = set()
    
    while _time_ok():
        all_dist = _compute_all_distances(adj_masks, n)
        bad_pairs = _get_bad_pairs(all_dist, n)
        if not bad_pairs:
            break
        
        best_edge = None
        best_score = None
        best_gain = -1
        best_unique = -1
        
        for i, j in candidates:
            if (i, j) in added_set:
                continue
            # Triangle-free check with current graph
            if (adj_masks[i] & adj_masks[j]) != 0:
                continue
            
            # Compute which bad pairs get fixed
            dist_i = all_dist[i]
            dist_j = all_dist[j]
            gain = 0
            fixed_pairs = set()
            for (u, v) in bad_pairs:
                # New distance via (i,j)
                nd1 = dist_i[u] + 1 + dist_j[v]
                nd2 = dist_i[v] + 1 + dist_j[u]
                nd3 = dist_j[u] + 1 + dist_i[v]
                nd4 = dist_j[v] + 1 + dist_i[u]
                new_d = min(all_dist[u][v], nd1, nd2, nd3, nd4)
                if new_d <= 4:
                    gain += 1
                    fixed_pairs.add((u, v))
            
            if gain == 0:
                continue
            
            # Score: (gain, unique coverage) — prefer edges fixing pairs not fixed by others
            # For simplicity, use gain as primary, then a heuristic for uniqueness
            # Count how many of the fixed pairs are "rare" (few other edges can fix them)
            # Approximate: prefer larger gain, tie-break by sum of pair indices (deterministic)
            if gain > best_gain:
                best_gain = gain
                best_edge = (i, j)
                best_unique = sum(fixed_pairs)
            elif gain == best_gain and best_edge is not None:
                # Tie-break: prefer edge whose fixed pairs have higher total "difficulty"
                # (larger indices = further from center, harder to fix otherwise)
                u_sum = sum(u + v for (u, v) in fixed_pairs)
                if u_sum > best_unique:
                    best_edge = (i, j)
                    best_unique = u_sum
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        added_set.add((u, v))
        added.append([u, v])
    
    return added

def _local_search_minimize(n, base_edges, added_edges):
    added = list(added_edges)
    improved = True
    while improved and _time_ok():
        improved = False
        for i in range(len(added) - 1, -1, -1):
            if not _time_ok():
                break
            trial = added[:i] + added[i + 1:]
            if _is_valid(n, base_edges, trial):
                added = trial
                improved = True
                break  # restart scan after removal
    return added

def construct():
    n = P['n']
    base = P['edges']
    
    # 1. Greedy construction
    added = _greedy_construct(n, base)
    
    # 2. Local search to minimize
    added = _local_search_minimize(n, base, added)
    
    # 3. Verify and return
    if _is_valid(n, base, added):
        return [[int(u), int(v)] for u, v in added]
    
    # Fallback: dense bipartite + prune
    adj_masks, base_set = _build_adj_masks(n, base)
    dense = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in base_set:
                continue
            if i % 2 == j % 2:
                continue
            if (adj_masks[i] & adj_masks[j]) == 0:
                dense.append([i, j])
    
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(dense) - 1, -1, -1):
            if not _time_ok():
                break
            trial = dense[:i] + dense[i + 1:]
            if _is_valid(n, base, trial):
                dense = trial
                changed = True
                break
    
    if _is_valid(n, base, dense):
        return [[int(u), int(v)] for u, v in dense]
    
    return []

# EVOLVE-BLOCK-END
