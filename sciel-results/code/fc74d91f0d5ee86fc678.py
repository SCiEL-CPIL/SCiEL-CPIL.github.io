"""Longest path in G(100, 1.8/100) pinned graph - combined backtracking with randomized restarts."""
import time
import random
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute adjacency lists and degrees
    adj_list = [sorted(adj[i]) for i in range(n)]
    deg = [len(adj[i]) for i in range(n)]
    
    best_path = []
    best_len = 0
    
    def update_best(path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    def get_neighbors_heuristic(current, seen, rng=None):
        """Get unvisited neighbors sorted by Warnsdorff-like heuristic."""
        neighbors = []
        for u in adj_list[current]:
            if u not in seen:
                # Count unvisited neighbors of u (excluding current)
                unvisited_count = sum(1 for w in adj_list[u] if w not in seen and w != current)
                if rng is not None:
                    perturbation = rng.random() * 0.1
                    neighbors.append((unvisited_count + perturbation, deg[u], u))
                else:
                    neighbors.append((unvisited_count, deg[u], u))
        neighbors.sort()
        return [u for _, _, u in neighbors]
    
    def backtrack(path, seen, rng=None):
        """Backtracking search with time budget."""
        nonlocal best_path, best_len, deadline
        
        current_len = len(path)
        if current_len > best_len:
            best_len = current_len
            best_path = list(path)
        
        current = path[-1]
        remaining = n - len(seen)
        
        # Pruning: can we beat best?
        if current_len + remaining <= best_len:
            return
        
        neighbors = get_neighbors_heuristic(current, seen, rng)
        if not neighbors:
            return
        
        for v in neighbors:
            if time.time() >= deadline:
                return
            path.append(v)
            seen.add(v)
            backtrack(path, seen, rng)
            path.pop()
            seen.discard(v)
            if time.time() >= deadline:
                return
    
    # Phase 1: Deterministic backtracking from each vertex
    for start in range(n):
        if time.time() >= deadline:
            break
        path = [start]
        seen = {start}
        backtrack(path, seen)
    
    # Phase 2: Randomized restarts with different heuristics
    rng = random.Random(42)
    for restart in range(500):
        if time.time() >= deadline:
            break
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        backtrack(path, seen, rng)
    
    # Phase 3: Greedy extensions from best path endpoints
    if best_path and time.time() < deadline:
        bp = list(best_path)
        bs = set(bp)
        
        # Try extending from the front
        for v in adj[bp[0]] - bs:
            if time.time() >= deadline:
                break
            new_path = [v] + bp
            new_seen = bs | {v}
            # Try to extend further greedily
            while adj[new_path[-1]] - new_seen:
                candidates = adj[new_path[-1]] - new_seen
                w = min(candidates, key=lambda x: (len(adj[x] - new_seen), x))
                new_path.append(w)
                new_seen.add(w)
            update_best(new_path)
        
        # Try extending from the back
        for v in adj[bp[-1]] - bs:
            if time.time() >= deadline:
                break
            new_path = bp + [v]
            new_seen = bs | {v}
            while adj[new_path[-1]] - new_seen:
                candidates = adj[new_path[-1]] - new_seen
                w = min(candidates, key=lambda x: (len(adj[x] - new_seen), x))
                new_path.append(w)
                new_seen.add(w)
            update_best(new_path)
    
    # Phase 4: Try multiple greedy strategies from high-degree starts
    degree_sorted = sorted(range(n), key=lambda x: -deg[x])
    for start in degree_sorted[:20]:
        if time.time() >= deadline:
            break
        
        strategies = [
            lambda cands, s: min(cands, key=lambda w: (len(adj[w] - s), w)),
            lambda cands, s: max(cands, key=lambda w: (len(adj[w] - s), w)),
            lambda cands, s: min(cands, key=lambda w: w),
            lambda cands, s: max(cands, key=lambda w: w),
        ]
        
        for strat in strategies:
            if time.time() >= deadline:
                break
            path = [start]
            seen = {start}
            while adj[path[-1]] - seen:
                candidates = adj[path[-1]] - seen
                v = strat(candidates, seen)
                path.append(v)
                seen.add(v)
            update_best(path)
    
    # Phase 5: More backtracking from promising starts if time remains
    if time.time() < deadline - 10:
        for start in degree_sorted[:15]:
            if time.time() >= deadline:
                break
            path = [start]
            seen = {start}
            backtrack(path, seen)
    
    return best_path if best_path else [0]
