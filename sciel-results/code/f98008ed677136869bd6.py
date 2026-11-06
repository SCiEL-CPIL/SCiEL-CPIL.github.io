"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_path(adj, start, seen, rng, mode=0):
    """Greedy path extension with different heuristics.
    mode 0: prefer neighbor with fewest unvisited neighbors (dead-end avoidance)
    mode 1: prefer neighbor with most unvisited neighbors (expansion)
    mode 2: random among candidates
    """
    path = [start]
    seen = seen | {start}
    while True:
        cur = path[-1]
        cands = adj[cur] - seen
        if not cands:
            break
        if mode == 0:
            # Prefer neighbor with fewest remaining options (avoid dead ends)
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        elif mode == 1:
            # Prefer neighbor with most remaining options
            v = max(cands, key=lambda w: (len(adj[w] - seen), w))
        else:
            v = rng.choice(sorted(cands))
        path.append(v)
        seen.add(v)
    return path

def _try_extend_both_ends(adj, path):
    """Try to extend path at both ends."""
    path = list(path)
    changed = True
    while changed:
        changed = False
        # Extend at start
        if path:
            cur = path[0]
            cands = adj[cur] - set(path)
            if cands:
                path.insert(0, min(cands, key=lambda w: (len(adj[w] - set(path)), w)))
                changed = True
        # Extend at end
        if path:
            cur = path[-1]
            cands = adj[cur] - set(path)
            if cands:
                path.append(min(cands, key=lambda w: (len(adj[w] - set(path)), w)))
                changed = True
    return path

def _dfs_longest(adj, start, seen, depth, best_path, deadline, max_depth=200):
    """DFS to find longest path from start, with pruning."""
    if time.time() > deadline:
        return
    if depth > len(best_path):
        best_path.clear()
        best_path.extend(seen)
    if depth >= max_depth:
        return
    cur = list(seen)[-1] if seen else start
    cands = sorted(adj[cur] - seen)
    for v in cands:
        if time.time() > deadline:
            return
        seen.add(v)
        _dfs_longest(adj, v, seen, depth + 1, best_path, deadline, max_depth)
        seen.discard(v)

def _randomized_search(adj, rng, deadline, n_iter=500):
    """Randomized multi-start search with different heuristics."""
    best = []
    n = P['n']
    modes = [0, 1, 2]
    for i in range(n_iter):
        if time.time() > deadline:
            break
        start = rng.randrange(n)
        mode = rng.choice(modes)
        path = _greedy_path(adj, start, set(), rng, mode)
        path = _try_extend_both_ends(adj, path)
        if len(path) > len(best):
            best = list(path)
    return best

def _local_improve(adj, path, deadline, n_iter=200):
    """Try to improve path by swapping/reordering via 2-opt style moves."""
    path = list(path)
    n = P['n']
    best_len = len(path)
    rng = _random.Random(42)
    for _ in range(n_iter):
        if time.time() > deadline:
            break
        cur = list(path)
        # Try inserting a vertex not in path
        in_path = set(cur)
        for v in range(n):
            if v in in_path:
                continue
            # Find best position to insert v
            best_pos = -1
            best_key = -1
            for i in range(len(cur)):
                prev = cur[i-1] if i > 0 else None
                nxt = cur[i] if i < len(cur) else None
                ok = True
                if prev is not None and v not in adj[prev]:
                    ok = False
                if nxt is not None and v not in adj[nxt]:
                    ok = False
                if ok:
                    # Score: prefer positions where v connects well
                    key = 0
                    if prev is not None:
                        key += len(adj[v] - in_path)
                    if nxt is not None:
                        key += len(adj[v] - in_path)
                    if key > best_key:
                        best_key = key
                        best_pos = i
            if best_pos >= 0:
                cur.insert(best_pos, v)
                if len(cur) > best_len:
                    best_len = len(cur)
                    path = list(cur)
                break
    return path

def _component_paths(adj):
    """Find connected components and compute longest path in each."""
    n = P['n']
    visited = set()
    components = []
    for i in range(n):
        if i in visited:
            continue
        # BFS to find component
        comp = []
        stack = [i]
        visited.add(i)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if w not in visited:
                    visited.add(w)
                    stack.append(w)
        components.append(comp)
    return components

def _longest_in_component(adj, comp, deadline):
    """Find longest path within a component using multi-strategy."""
    if not comp:
        return []
    best = []
    comp_set = set(comp)
    
    # Multi-start greedy
    for start in comp:
        if time.time() > deadline:
            break
        for mode in range(3):
            path = _greedy_path(adj, start, set(), _random.Random(start*3+mode), mode)
            path = _try_extend_both_ends(adj, path)
            if len(path) > len(best):
                best = list(path)
    
    # DFS from promising starts (low-degree vertices tend to be endpoints)
    sorted_comp = sorted(comp, key=lambda v: len(adj[v] & comp_set))
    for start in sorted_comp[:10]:
        if time.time() > deadline:
            break
        best_path = list(best)
        _dfs_longest(adj, start, {start}, 1, best_path, deadline, max_depth=len(comp))
        if len(best_path) > len(best):
            best = list(best_path)
    
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # 10s safety margin
    
    adj = _build_adj()
    n = P['n']
    rng = _random.Random(748)
    
    # Strategy 1: Per-component longest path
    components = _component_paths(adj)
    best = []
    for comp in components:
        if time.time() > deadline:
            break
        cp = _longest_in_component(adj, comp, deadline)
        if len(cp) > len(best):
            best = list(cp)
    
    # Strategy 2: Global randomized multi-start
    if time.time() < deadline - 5.0:
        rp = _randomized_search(adj, rng, deadline, n_iter=1000)
        if len(rp) > len(best):
            best = list(rp)
    
    # Strategy 3: Local improvement
    if time.time() < deadline - 2.0 and best:
        improved = _local_improve(adj, best, deadline, n_iter=500)
        if len(improved) > len(best):
            best = improved
    
    # Strategy 4: More randomized restarts if time remains
    if time.time() < deadline - 1.0:
        for i in range(2000):
            if time.time() > deadline:
                break
            start = rng.randrange(n)
            mode = rng.choice([0, 1, 2])
            path = _greedy_path(adj, start, set(), rng, mode)
            path = _try_extend_both_ends(adj, path)
            if len(path) > len(best):
                best = list(path)
    
    # Final local improvement
    if time.time() < deadline - 0.5 and best:
        improved = _local_improve(adj, best, deadline, n_iter=200)
        if len(improved) > len(best):
            best = improved
    
    # Safety: ensure valid path
    if not best:
        best = [0]
    
    # Validate
    in_path = set(best)
    if len(in_path) != len(best):
        # Remove duplicates
        seen = set()
        clean = []
        for v in best:
            if v not in seen:
                seen.add(v)
                clean.append(v)
        best = clean
    # Verify edges
    valid = []
    for i, v in enumerate(best):
        if i == 0:
            valid.append(v)
        else:
            if v in adj[valid[-1]]:
                valid.append(v)
            else:
                break
    best = valid if valid else [best[0]]
    
    return best
# EVOLVE-BLOCK-END
