"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

import time
import random
from collections import deque

def _build_adj():
    a = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        a[u].add(v)
        a[v].add(u)
    return a

def _greedy_path(adj, start, strategy, rng=None):
    """Greedy path extension with different neighbor-selection strategies."""
    path = [start]
    seen = {start}
    while True:
        cands = adj[path[-1]] - seen
        if not cands:
            break
        if strategy == 'min_deg':
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_deg':
            v = max(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'random':
            v = rng.choice(sorted(cands))
        elif strategy == 'weighted':
            # Prefer neighbors with fewer unvisited neighbors (dead-end avoidance)
            weights = [1.0 / max(1, len(adj[w] - seen)) for w in cands]
            v = sorted(cands)[weights.index(max(weights))]
        else:
            v = min(cands)
        path.append(v)
        seen.add(v)
    return path

def _backtrack_search(adj, start, max_depth, rng, deadline):
    """DFS with backtracking, limited by depth and time."""
    best_path = [start]
    
    def dfs(path, seen, depth):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = path[:]
        if depth >= max_depth:
            return
        cands = sorted(adj[path[-1]] - seen)
        rng.shuffle(cands)
        for v in cands:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen, depth + 1)
            path.pop()
            seen.discard(v)
    
    dfs([start], {start}, 0)
    return best_path

def _two_end_grow(adj, start1, start2, rng, deadline):
    """Grow path from two ends inward."""
    if start1 == start2:
        return [start1]
    path1 = [start1]
    path2 = [start2]
    seen = {start1, start2}
    
    while True:
        # Try to extend end 1
        cands1 = adj[path1[-1]] - seen
        if cands1:
            v = rng.choice(sorted(cands1))
            path1.append(v)
            seen.add(v)
        else:
            # Try to extend end 2
            cands2 = adj[path2[-1]] - seen
            if cands2:
                v = rng.choice(sorted(cands2))
                path2.append(v)
                seen.add(v)
            else:
                break
        if time.time() > deadline:
            break
    
    # Combine: path1 forward + path2 reversed
    result = path1 + path2[::-1]
    # Remove duplicates if any (shouldn't happen but safety)
    seen_r = set()
    unique = []
    for x in result:
        if x not in seen_r:
            unique.append(x)
            seen_r.add(x)
    return unique

def construct():
    deadline = time.time() + 85  # 5s safety margin
    adj = _build_adj()
    best = []
    
    # Strategy 1: Deterministic greedy from every start
    for start in range(P['n']):
        path = _greedy_path(adj, start, 'min_deg')
        if len(path) > len(best):
            best = path
    
    # Strategy 2: Randomized restarts with different heuristics
    rng = random.Random(42)
    strategies = ['min_deg', 'max_deg', 'random', 'weighted']
    
    for strat in strategies:
        for start in range(P['n']):
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, strat, rng)
            if len(path) > len(best):
                best = path
    
    # Strategy 3: Backtracking search from promising starts
    # Focus on high-degree vertices as potential path cores
    degs = [(len(adj[i]), i) for i in range(P['n'])]
    degs.sort(reverse=True)
    
    for deg, start in degs[:20]:
        if time.time() > deadline:
            break
        max_depth = min(P['n'], 30)  # Limit depth to keep it tractable
        path = _backtrack_search(adj, start, max_depth, rng, deadline)
        if len(path) > len(best):
            best = path
    
    # Strategy 4: Two-end growth from pairs of high-degree vertices
    top_verts = [v for _, v in degs[:15]]
    for i in range(len(top_verts)):
        for j in range(i + 1, len(top_verts)):
            if time.time() > deadline:
                break
            path = _two_end_grow(adj, top_verts[i], top_verts[j], rng, deadline)
            if len(path) > len(best):
                best = path
    
    # Final: try all starts with random strategy one more time if time permits
    for start in range(P['n']):
        if time.time() > deadline:
            break
        for _ in range(3):
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, 'random', rng)
            if len(path) > len(best):
                best = path
    
    return best
# EVOLVE-BLOCK-END
