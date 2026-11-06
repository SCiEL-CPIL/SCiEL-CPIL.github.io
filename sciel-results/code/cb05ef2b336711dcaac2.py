"""Longest path in G(100, 1.8/100) pinned graph - backtracking with time budget."""
import time
import sys
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute adjacency as lists for faster iteration
    adj_list = [list(s) for s in adj]
    
    deadline = time.time() + 85  # safety margin before 90s kill
    
    best_path = []
    best_len = 0
    
    def greedy_extend(path, seen):
        """Greedy extension from current path endpoint."""
        path = list(path)
        seen = set(seen)
        while adj[path[-1]] - seen:
            # Choose neighbor with fewest unvisited neighbors (dead-end first)
            candidates = adj[path[-1]] - seen
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        return path
    
    # Phase 1: Greedy from every start, try both directions
    for start in range(n):
        if time.time() > deadline:
            break
        for direction in range(2):
            path = [start]
            seen = {start}
            while adj[path[-1]] - seen:
                candidates = adj[path[-1]] - seen
                if direction == 0:
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                else:
                    v = max(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
    
    # Phase 2: Backtracking search with time budget
    # Try to extend the best path found by backtracking
    def backtrack(path, seen, depth, max_depth):
        """Backtracking to find longer paths."""
        nonlocal best_path, best_len, deadline
        
        if time.time() > deadline:
            return False
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        if len(path) == n:
            return True
        
        current = path[-1]
        candidates = sorted(adj[current] - seen, key=lambda w: len(adj[w] - seen))
        
        for v in candidates:
            if time.time() > deadline:
                return False
            path.append(v)
            seen.add(v)
            backtrack(path, seen, depth + 1, max_depth)
            path.pop()
            seen.discard(v)
            
            if time.time() > deadline:
                return False
        
        return False
    
    # Phase 3: More aggressive search - try to build long paths from various starts
    # Use a smarter heuristic: try to find paths that visit high-degree nodes
    degree = [len(adj[i]) for i in range(n)]
    
    # Sort starts by degree descending
    starts_by_degree = sorted(range(n), key=lambda x: -degree[x])
    
    for start in starts_by_degree[:30]:  # Try top 30 degree starts
        if time.time() > deadline:
            break
        
        # Try multiple greedy strategies from this start
        strategies = [
            lambda cands, s: min(cands, key=lambda w: (len(adj[w] - s), w)),
            lambda cands, s: max(cands, key=lambda w: (len(adj[w] - s), w)),
            lambda cands, s: min(cands, key=lambda w: w),
            lambda cands, s: max(cands, key=lambda w: w),
        ]
        
        for strat in strategies:
            if time.time() > deadline:
                break
            path = [start]
            seen = {start}
            while adj[path[-1]] - seen:
                candidates = adj[path[-1]] - seen
                v = strat(candidates, seen)
                path.append(v)
                seen.add(v)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
    
    # Phase 4: Local search - try to improve best_path by swapping/extending
    if best_path and time.time() < deadline:
        # Try extending from both ends of best_path
        bp = list(best_path)
        bs = set(bp)
        
        # Try extending from the front
        for v in adj[bp[0]] - bs:
            if time.time() > deadline:
                break
            new_path = [v] + bp
            new_seen = bs | {v}
            # Try to extend further
            while adj[new_path[-1]] - new_seen:
                candidates = adj[new_path[-1]] - new_seen
                w = min(candidates, key=lambda x: (len(adj[x] - new_seen), x))
                new_path.append(w)
                new_seen.add(w)
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = list(new_path)
        
        # Try extending from the back
        for v in adj[bp[-1]] - bs:
            if time.time() > deadline:
                break
            new_path = bp + [v]
            new_seen = bs | {v}
            while adj[new_path[-1]] - new_seen:
                candidates = adj[new_path[-1]] - new_seen
                w = min(candidates, key=lambda x: (len(adj[x] - new_seen), x))
                new_path.append(w)
                new_seen.add(w)
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = list(new_path)
    
    # Phase 5: Iterative deepening backtracking from promising starts
    if time.time() < deadline:
        for start in starts_by_degree[:10]:
            if time.time() > deadline:
                break
            path = [start]
            seen = {start}
            # Limited backtracking
            backtrack(path, seen, 0, 100)
            if time.time() > deadline:
                break
    
    # Phase 6: Try to find Hamiltonian-like paths using more search
    if time.time() < deadline - 5:
        # Use a different approach: try to connect components or extend
        # Try all pairs of consecutive vertices as potential path segments
        for i in range(n):
            if time.time() > deadline:
                break
            for j in adj[i]:
                if time.time() > deadline:
                    break
                # Build path starting i->j
                path = [i, j]
                seen = {i, j}
                while adj[path[-1]] - seen:
                    candidates = adj[path[-1]] - seen
                    v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                    path.append(v)
                    seen.add(v)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
    
    return best_path if best_path else [0]
