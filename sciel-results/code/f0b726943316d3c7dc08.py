"""Rewritten program: branch-and-bound longest path with multiple strategies."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_path(adj, start, strategy='min_degree'):
    """Greedy path extension from start using a given strategy."""
    n = P['n']
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = [v for v in adj[cur] if v not in seen]
        if not candidates:
            break
        if strategy == 'min_degree':
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_degree':
            v = max(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'min_remaining':
            # Choose vertex with fewest unvisited neighbors (to avoid trapping)
            v = min(candidates, key=lambda w: len(adj[w] - seen))
        else:
            v = candidates[0]
        path.append(v)
        seen.add(v)
    return path

def _dfs_bounded(adj, start, max_depth, time_limit, best_path, best_len):
    """DFS with depth limit and time check. Returns updated best_path, best_len."""
    n = P['n']
    path = [start]
    seen = {start}
    
    def dfs():
        nonlocal best_path, best_len
        if time.time() > time_limit:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        if len(path) >= max_depth:
            return
        cur = path[-1]
        # Order neighbors by degree (heuristic)
        neighbors = [v for v in adj[cur] if v not in seen]
        neighbors.sort(key=lambda w: len(adj[w]))
        for v in neighbors:
            if time.time() > time_limit:
                return
            path.append(v)
            seen.add(v)
            dfs()
            path.pop()
            seen.remove(v)
    
    dfs()
    return best_path, best_len

def _random_walks(adj, num_walks, time_limit, best_path, best_len):
    """Try random greedy walks from random starts."""
    import random
    rng = random.Random(42)
    n = P['n']
    for _ in range(num_walks):
        if time.time() > time_limit:
            break
        start = rng.randint(0, n - 1)
        path = _greedy_path(adj, start, strategy='min_degree')
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    return best_path, best_len

def _improve_path(adj, path, time_limit):
    """Try to extend or rearrange a path to make it longer."""
    n = P['n']
    best = list(path)
    best_len = len(best)
    
    # Try extending from both ends
    for _ in range(50):  # limited attempts
        if time.time() > time_limit:
            break
        # Try extending from the end
        cur = best[-1]
        seen = set(best)
        candidates = [v for v in adj[cur] if v not in seen]
        if candidates:
            v = min(candidates, key=lambda w: len(adj[w] - seen))
            new_path = best + [v]
            if len(new_path) > best_len:
                best = new_path
                best_len = len(best)
                continue
        # Try extending from the front
        cur = best[0]
        seen = set(best)
        candidates = [v for v in adj[cur] if v not in seen]
        if candidates:
            v = min(candidates, key=lambda w: len(adj[w] - seen))
            new_path = [v] + best
            if len(new_path) > best_len:
                best = new_path
                best_len = len(best)
                continue
        break
    
    return best, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    adj = _build_adj()
    n = P['n']
    
    # Phase 1: Greedy from all starts with multiple strategies
    best_path = []
    best_len = 0
    for start in range(n):
        if time.time() > deadline:
            break
        for strat in ['min_degree', 'max_degree']:
            path = _greedy_path(adj, start, strategy=strat)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
    
    # Phase 2: Random walks
    remaining = deadline - time.time()
    if remaining > 1:
        best_path, best_len = _random_walks(adj, 200, deadline, best_path, best_len)
    
    # Phase 3: DFS from promising starts
    remaining = deadline - time.time()
    if remaining > 2:
        # Sort starts by degree, try high-degree first
        starts_sorted = sorted(range(n), key=lambda v: -len(adj[v]))
        for start in starts_sorted[:20]:
            if time.time() > deadline:
                break
            # DFS with depth limit = current best + 10
            max_depth = min(best_len + 15, n)
            best_path, best_len = _dfs_bounded(adj, start, max_depth, deadline, best_path, best_len)
    
    # Phase 4: Try to improve the best path found
    remaining = deadline - time.time()
    if remaining > 1 and len(best_path) > 0:
        best_path, best_len = _improve_path(adj, best_path, deadline)
    
    # Final fallback: if no path found, return a single vertex
    if not best_path:
        best_path = [0]
    
    # Verify the path is valid
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
            # Invalid path, truncate to valid prefix
            valid = [best_path[0]]
            for v in best_path[1:]:
                if v in adj[valid[-1]] and v not in valid:
                    valid.append(v)
                else:
                    break
            best_path = valid
            if not best_path:
                best_path = [0]
            break
    
    return best_path
# EVOLVE-BLOCK-END
