"""Rewritten program: stronger longest-path search with 2-opt, DFS, and multiple heuristics."""
import time
import random as _rnd

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute adjacency as frozensets for fast operations
    adj_frozen = [frozenset(s) for s in adj]
    
    deadline = time.time() + 55  # 55s safety margin
    
    def is_edge(u, v):
        return v in adj[u]
    
    def greedy_path(start, strategy='min', rng=None):
        """Greedy path construction from start vertex."""
        path = [start]
        seen = {start}
        while True:
            last = path[-1]
            candidates = adj[last] - seen
            if not candidates:
                break
            if strategy == 'min':
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            elif strategy == 'max':
                v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
            elif strategy == 'random':
                v = rng.choice(sorted(candidates))
            else:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        return path
    
    def two_opt_refine(path, max_iter=200):
        """Apply 2-opt style reversals to extend the path.
        If path[i] and path[j] are connected (j-i>=2), reverse path[i+1..j]
        to potentially expose new endpoints."""
        path = list(path)
        seen = set(path)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            # Try all pairs (i, j) where i < j-1 and edge exists between path[i] and path[j]
            # Reversing path[i+1..j] keeps path valid and may change endpoints
            for i in range(len(path) - 2):
                for j in range(i + 2, len(path)):
                    if is_edge(path[i], path[j]):
                        # Reverse path[i+1..j]
                        new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                        # Check if new endpoints can be extended
                        new_seen = set(new_path)
                        # Check if last vertex has unvisited neighbors
                        last = new_path[-1]
                        first = new_path[0]
                        can_extend_last = bool(adj[last] - new_seen)
                        can_extend_first = bool(adj[first] - new_seen)
                        if can_extend_last or can_extend_first:
                            path = new_path
                            improved = True
                            break
                if improved:
                    break
        # After 2-opt, try to extend both ends greedily
        # Extend the end with more options
        seen = set(path)
        # Extend forward (from last)
        while True:
            last = path[-1]
            cands = adj[last] - seen
            if not cands:
                break
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        # Extend backward (from first)
        while True:
            first = path[0]
            cands = adj[first] - seen
            if not cands:
                break
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
        return path
    
    def try_extend_both_ends(path):
        """Greedily extend both ends of the path."""
        path = list(path)
        seen = set(path)
        changed = True
        while changed:
            changed = False
            # Extend from last
            while True:
                last = path[-1]
                cands = adj[last] - seen
                if not cands:
                    break
                v = min(cands, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
                changed = True
            # Extend from first
            while True:
                first = path[0]
                cands = adj[first] - seen
                if not cands:
                    break
                v = min(cands, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        return path
    
    def dfs_longest_on_component(start, visited_set, depth_limit=12):
        """DFS to find longest path starting from start, avoiding visited_set.
        Returns the path (list of vertices) of maximum length found."""
        best = [start]
        seen = set(visited_set)
        seen.add(start)
        
        def dfs(path, seen):
            nonlocal best
            if len(path) > len(best):
                best = list(path)
            if len(path) >= depth_limit:
                return
            last = path[-1]
            # Order neighbors by fewest unvisited options (to prune)
            cands = [w for w in adj[last] if w not in seen]
            cands.sort(key=lambda w: (len(adj[w] - seen), w))
            for w in cands:
                if w in seen:
                    continue
                seen.add(w)
                path.append(w)
                dfs(path, seen)
                path.pop()
                seen.discard(w)
        
        dfs([start], seen)
        return best
    
    # Main search
    best_path = []
    
    # Phase 1: Greedy from all starts with multiple strategies
    strategies = ['min', 'max']
    for start in range(n):
        if time.time() > deadline:
            break
        for strat in strategies:
            path = greedy_path(start, strat)
            if len(path) > len(best_path):
                best_path = path
    
    # Phase 2: Random greedy with 2-opt
    rng = _rnd.Random(42)
    for start in range(n):
        if time.time() > deadline:
            break
        for _ in range(3):
            path = greedy_path(start, 'random', rng)
            path = two_opt_refine(path)
            path = try_extend_both_ends(path)
            if len(path) > len(best_path):
                best_path = path
    
    # Phase 3: 2-opt on best path found so far
    if best_path:
        refined = two_opt_refine(best_path, max_iter=500)
        refined = try_extend_both_ends(refined)
        if len(refined) > len(best_path):
            best_path = refined
    
    # Phase 4: Try to extend best_path using DFS on remaining component
    if best_path and time.time() < deadline:
        seen_in_best = set(best_path)
        # Find unvisited vertices that are adjacent to path endpoints
        for endpoint in [best_path[0], best_path[-1]]:
            if time.time() > deadline:
                break
            cands = [w for w in adj[endpoint] if w not in seen_in_best]
            if cands:
                # Try DFS from each candidate
                for c in cands:
                    if time.time() > deadline:
                        break
                    ext = dfs_longest_on_component(c, seen_in_best, depth_limit=10)
                    if len(ext) > 1:
                        # Prepend or append
                        new_path = best_path + ext if endpoint == best_path[-1] else ext + best_path
                        if len(new_path) > len(best_path):
                            best_path = new_path
                            seen_in_best = set(best_path)
    
    # Phase 5: More 2-opt iterations on the current best
    if best_path and time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            refined = two_opt_refine(best_path, max_iter=200)
            refined = try_extend_both_ends(refined)
            if len(refined) > len(best_path):
                best_path = refined
            else:
                break
    
    # Phase 6: Try all starts again with the improved best as seed
    if best_path and time.time() < deadline:
        # Try to find a path that includes more of the graph
        # Use the complement: try to build a path through unvisited vertices
        seen_in_best = set(best_path)
        unvisited = [v for v in range(n) if v not in seen_in_best]
        if unvisited:
            # Find connected components of unvisited subgraph
            visited_comp = set()
            for u in unvisited:
                if u in visited_comp:
                    continue
                # BFS to find component
                comp = [u]
                visited_comp.add(u)
                queue = [u]
                while queue:
                    x = queue.pop()
                    for y in adj[x]:
                        if y not in seen_in_best and y not in visited_comp:
                            visited_comp.add(y)
                            comp.append(y)
                            queue.append(y)
                # If component is small enough, try DFS
                if len(comp) <= 15 and time.time() < deadline:
                    for start_c in comp:
                        ext = dfs_longest_on_component(start_c, seen_in_best, depth_limit=len(comp))
                        if len(ext) > 1:
                            # Check if ext can connect to best_path
                            # Try prepending
                            if ext and is_edge(ext[-1], best_path[0]):
                                new_path = ext + best_path
                                if len(new_path) > len(best_path):
                                    best_path = new_path
                            # Try appending
                            if ext and is_edge(ext[0], best_path[-1]):
                                new_path = best_path + ext
                                if len(new_path) > len(best_path):
                                    best_path = new_path
    
    # Final: ensure we have a valid path
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
