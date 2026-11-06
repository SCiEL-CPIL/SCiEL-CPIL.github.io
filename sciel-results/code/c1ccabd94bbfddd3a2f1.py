"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

import time
import random
import sys
from collections import deque

def _build_adj():
    a = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        a[u].add(v)
        a[v].add(u)
    return a

def _find_largest_component(adj):
    visited = set()
    components = []
    for start in range(P['n']):
        if start not in visited:
            comp = []
            stack = [start]
            visited.add(start)
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in adj[v]:
                    if u not in visited:
                        visited.add(u)
                        stack.append(u)
            components.append(comp)
    return max(components, key=len) if components else []

def _greedy_path(adj, start, strategy, rng=None, deadline=None):
    """Greedy path extension with different neighbor-selection strategies."""
    path = [start]
    seen = {start}
    while True:
        if deadline and time.time() > deadline:
            break
        cands = adj[path[-1]] - seen
        if not cands:
            break
        if strategy == 'min_deg':
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_deg':
            v = max(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'random':
            v = rng.choice(sorted(cands))
        elif strategy == 'slack_aware':
            def slack_score(w):
                slack = len(adj[w] - seen)
                return (slack, len(adj[w]), w)
            v = min(cands, key=slack_score)
        elif strategy == 'endpoint_aware':
            v = min(cands, key=lambda w: (len(adj[w]), len(adj[w] - seen), w))
        else:
            v = min(cands)
        path.append(v)
        seen.add(v)
    return path

def _extend_from_both_ends(adj, path, deadline):
    """Try to extend a path from both ends using min-degree heuristic."""
    if not path:
        return path
    result = list(path)
    seen = set(result)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cands = adj[result[-1]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            result.append(v)
            seen.add(v)
            improved = True
        cands = adj[result[0]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            result.insert(0, v)
            seen.add(v)
            improved = True
    return result

def _two_end_grow(adj, start1, start2, rng, deadline):
    """Grow path from two ends inward."""
    if start1 == start2:
        return [start1]
    path1 = [start1]
    path2 = [start2]
    seen = {start1, start2}
    while True:
        if time.time() > deadline:
            break
        cands1 = adj[path1[-1]] - seen
        if cands1:
            v = min(cands1, key=lambda w: len(adj[w] - seen))
            path1.append(v)
            seen.add(v)
        else:
            cands2 = adj[path2[-1]] - seen
            if cands2:
                v = min(cands2, key=lambda w: len(adj[w] - seen))
                path2.append(v)
                seen.add(v)
            else:
                break
    result = path1 + path2[::-1]
    seen_r = set()
    unique = []
    for x in result:
        if x not in seen_r:
            unique.append(x)
            seen_r.add(x)
    return unique

def _backtrack_search(adj, start, max_depth, rng, deadline, endpoint_hints=None):
    """DFS with backtracking, using matching-based endpoint hints to guide choices."""
    best_path = [start]
    n = P['n']
    endpoint_set = set(endpoint_hints) if endpoint_hints else set()
    sys.setrecursionlimit(10000)
    
    def dfs(path, seen, depth):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = path[:]
        if depth >= max_depth:
            return
        cands = sorted(adj[path[-1]] - seen)
        if not cands:
            return
        def score(w):
            slack = len(adj[w] - seen)
            is_endpoint = 1 if w in endpoint_set else 0
            progress = depth / max(max_depth, 1)
            endpoint_penalty = is_endpoint * (1.0 - progress) * 2
            return (slack + endpoint_penalty, w)
        cands.sort(key=score)
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

def _bfs_path(adj, start, deadline):
    """BFS to find a simple path (not necessarily longest, but fast)."""
    visited = {start}
    parent = {start: None}
    queue = deque([start])
    farthest = start
    while queue:
        if time.time() > deadline:
            break
        v = queue.popleft()
        farthest = v
        for u in adj[v]:
            if u not in visited:
                visited.add(u)
                parent[u] = v
                queue.append(u)
    # Reconstruct path from start to farthest
    path = [farthest]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def _random_walk_path(adj, start, steps, deadline):
    """Random walk that backtracks when stuck."""
    path = [start]
    seen = {start}
    for _ in range(steps):
        if time.time() > deadline:
            break
        cands = adj[path[-1]] - seen
        if not cands:
            break
        v = random.choice(list(cands))
        path.append(v)
        seen.add(v)
    return path

def _local_search_improve(adj, path, deadline, rng=None):
    """Try to improve a path by local swaps and insertions."""
    if len(path) < 2:
        return path
    best = list(path)
    best_len = len(best)
    seen = set(best)
    
    # Try inserting vertices between consecutive path vertices
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best) - 1):
            if time.time() > deadline:
                break
            a, b = best[i], best[i+1]
            # Find common neighbors of a and b not in seen
            common = (adj[a] & adj[b]) - seen
            if common:
                # Try inserting each common neighbor
                for w in sorted(common, key=lambda x: len(adj[x])):
                    new_path = best[:i+1] + [w] + best[i+1:]
                    new_seen = set(new_path)
                    # Verify it's a valid path
                    valid = True
                    for j in range(len(new_path) - 1):
                        if new_path[j+1] not in adj[new_path[j]]:
                            valid = False
                            break
                    if valid and len(new_path) > best_len:
                        best = new_path
                        best_len = len(best)
                        seen = new_seen
                        improved = True
                        break
        # Also try extending ends
        cands = adj[best[-1]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            best.append(v)
            seen.add(v)
            best_len = len(best)
            improved = True
        cands = adj[best[0]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            best.insert(0, v)
            seen.add(v)
            best_len = len(best)
            improved = True
    return best

def _deep_backtrack(adj, start, deadline, endpoint_hints=None):
    """
    Iterative deepening DFS with strong pruning.
    Uses a time budget and tries to go as deep as possible.
    """
    best_path = [start]
    endpoint_set = set(endpoint_hints) if endpoint_hints else set()
    n = P['n']
    
    # Use iterative deepening with increasing depth limits
    # Start with a reasonable depth and increase
    current_depth = 10
    max_possible = n
    
    while current_depth <= max_possible and time.time() < deadline:
        found_deeper = False
        
        def dfs(path, seen, depth, limit):
            nonlocal best_path, found_deeper
            if time.time() > deadline:
                return False
            if len(path) > len(best_path):
                best_path = path[:]
            if depth >= limit:
                return len(path) > len(best_path) - 1  # Prune if unlikely to beat best
            cands = adj[path[-1]] - seen
            if not cands:
                return False
            # Sort candidates: prefer low slack, deprioritize endpoints early
            def score(w):
                slack = len(adj[w] - seen)
                is_endpoint = 1 if w in endpoint_set else 0
                progress = depth / max(limit, 1)
                endpoint_penalty = is_endpoint * (1.0 - progress) * 1.5
                return (slack + endpoint_penalty, w)
            cands_list = sorted(cands, key=score)
            for v in cands_list:
                if time.time() > deadline:
                    return False
                path.append(v)
                seen.add(v)
                if dfs(path, seen, depth + 1, limit):
                    found_deeper = True
                    path.pop()
                    seen.discard(v)
                    return True  # Found a good path, stop early
                path.pop()
                seen.discard(v)
            return False
        
        if dfs([start], {start}, 0, current_depth):
            found_deeper = True
            current_depth += 2  # Increase depth limit
        else:
            current_depth += 1
            if current_depth > len(best_path) + 5:
                break  # Not making progress
    
    return best_path

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    adj = _build_adj()
    best = []
    
    # Find the largest component
    largest_comp = _find_largest_component(adj)
    if not largest_comp:
        return [0]
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    # Step 1: Quick greedy from every start
    for start in largest_comp:
        path = _greedy_path(adj, start, 'min_deg')
        if len(path) > len(best):
            best = path
    
    # Step 2: BFS paths from every start (fast, gives a baseline)
    for start in largest_comp:
        if time.time() > deadline:
            break
        path = _bfs_path(adj, start, deadline)
        if len(path) > len(best):
            best = path
    
    # Step 3: Randomized greedy with multiple strategies
    rng = random.Random(42)
    strategies = ['min_deg', 'max_deg', 'random', 'slack_aware', 'endpoint_aware']
    for strat in strategies:
        for start in largest_comp:
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, strat, rng)
            if len(path) > len(best):
                best = path
    
    # Step 4: Two-end growth from pairs
    degs = [(len(adj[i]), i) for i in largest_comp]
    degs.sort(reverse=True)
    low_deg = [i for _, i in degs[-20:]]  # Low degree vertices as potential endpoints
    for i in range(len(low_deg)):
        for j in range(i + 1, min(i + 15, len(low_deg))):
            if time.time() > deadline:
                break
            path = _two_end_grow(adj, low_deg[i], low_deg[j], rng, deadline)
            if len(path) > len(best):
                best = path
    
    # Step 5: Deep backtracking from promising starts
    # Start from low-degree vertices (likely endpoints)
    for deg, start in degs[-15:]:
        if time.time() > deadline:
            break
        path = _deep_backtrack(adj, start, deadline)
        if len(path) > len(best):
            best = path
    
    # Step 6: Local search improvement on best path
    if best and time.time() < deadline:
        improved = _local_search_improve(adj, best[:], deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    # Step 7: More randomized searches
    for _ in range(50):
        if time.time() > deadline:
            break
        start = rng.choice(largest_comp)
        strat = rng.choice(strategies)
        path = _greedy_path(adj, start, strat, rng)
        if len(path) > len(best):
            best = path
        # Also try extending from both ends
        if len(path) >= 3:
            extended = _extend_from_both_ends(adj, path[:], deadline)
            if len(extended) > len(best):
                best = extended
    
    # Step 8: Backtracking from high-degree vertices (internal path candidates)
    for deg, start in degs[:15]:
        if time.time() > deadline:
            break
        path = _deep_backtrack(adj, start, deadline)
        if len(path) > len(best):
            best = path
    
    # Step 9: Final local search polish
    if best and time.time() < deadline:
        improved = _local_search_improve(adj, best[:], deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    # Step 10: Final extension
    if best and time.time() < deadline:
        extended = _extend_from_both_ends(adj, best[:], deadline)
        if len(extended) > len(best):
            best = extended
    
    # Step 11: Try a few more deep backtracks with different random seeds
    for seed in [123, 456, 789]:
        if time.time() > deadline:
            break
        rng2 = random.Random(seed)
        for _ in range(5):
            if time.time() > deadline:
                break
            start = rng2.choice(largest_comp)
            path = _deep_backtrack(adj, start, deadline)
            if len(path) > len(best):
                best = path
    
    # Final local search
    if best and time.time() < deadline:
        improved = _local_search_improve(adj, best[:], deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    if not best:
        best = [largest_comp[0]]
    
    return best
# EVOLVE-BLOCK-END
