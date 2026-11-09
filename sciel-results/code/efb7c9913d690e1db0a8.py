"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to sorted lists for deterministic iteration
    adj_list = [sorted(lst) for lst in adj]
    
    best_path = [0]
    
    # Deadline guard: use 85 seconds to be safe (evaluator kills at 90s)
    deadline = time.time() + 85
    
    # Strategy 1: Greedy with multiple heuristics
    def greedy_path(start, heuristic):
        path = [start]
        seen = {start}
        while True:
            candidates = [v for v in adj_list[path[-1]] if v not in seen]
            if not candidates:
                break
            if heuristic == 0:
                # Min degree first (avoid dead ends)
                v = min(candidates, key=lambda w: (len(adj_list[w]) - len(seen & adj_list[w]), w))
            elif heuristic == 1:
                # Max remaining degree first
                v = max(candidates, key=lambda w: (len(adj_list[w]) - len(seen & adj_list[w]), -w))
            elif heuristic == 2:
                # Min degree, tie-break by vertex id
                v = min(candidates, key=lambda w: (len(adj_list[w]), w))
            elif heuristic == 3:
                # Random-ish: pick based on vertex id mod something
                v = candidates[0]
            else:
                v = candidates[0]
            path.append(v)
            seen.add(v)
        return path
    
    # Try greedy from every start with multiple heuristics
    for h in range(4):
        for start in range(n):
            if time.time() > deadline:
                break
            path = greedy_path(start, h)
            if len(path) > len(best_path):
                best_path = path
        if time.time() > deadline:
            break
    
    # Strategy 2: DFS-based search with backtracking for local improvements
    # Try to extend the best path found so far, or find longer paths via targeted DFS
    
    def try_extend(path):
        """Try to extend a path from either end."""
        path = list(path)
        extended = False
        # Try extending from the end
        while True:
            candidates = [v for v in adj_list[path[-1]] if v not in set(path)]
            if not candidates:
                break
            # Try each candidate
            for v in candidates:
                new_path = path + [v]
                # Check if we can extend further
                further = [w for w in adj_list[v] if w not in set(new_path)]
                if further:
                    # This extension is promising, commit it
                    path = new_path
                    extended = True
                    break
            else:
                # No candidate leads to further extension, try the one with max potential
                if candidates:
                    v = max(candidates, key=lambda w: len([x for x in adj_list[w] if x not in set(path)]))
                    path = path + [v]
                    extended = True
                    break
        # Try extending from the beginning
        while True:
            candidates = [v for v in adj_list[path[0]] if v not in set(path)]
            if not candidates:
                break
            for v in candidates:
                new_path = [v] + path
                further = [w for w in adj_list[v] if w not in set(new_path)]
                if further:
                    path = new_path
                    extended = True
                    break
            else:
                if candidates:
                    v = max(candidates, key=lambda w: len([x for x in adj_list[w] if x not in set(path)]))
                    path = [v] + path
                    extended = True
                    break
        return path
    
    # Extend the best path found
    best_path = try_extend(best_path)
    
    # Strategy 3: More exhaustive DFS from promising starts
    # Identify high-degree vertices as potential start points
    degree = [len(adj_list[i]) for i in range(n)]
    promising_starts = sorted(range(n), key=lambda i: -degree[i])
    
    def dfs_longest(current, path, seen, limit):
        """DFS to find longest path from current vertex."""
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = list(path)
        candidates = [v for v in adj_list[current] if v not in seen]
        if not candidates:
            return
        # Sort by degree descending (explore high-degree first)
        candidates.sort(key=lambda w: -degree[w])
        for v in candidates:
            if time.time() > deadline:
                return
            dfs_longest(v, path + [v], seen | {v}, limit)
    
    # Run DFS from top promising starts with a depth limit to avoid exponential blowup
    for start in promising_starts[:20]:
        if time.time() > deadline:
            break
        seen = {start}
        path = [start]
        dfs_longest(start, path, seen, 100)
    
    # Strategy 4: Local search - try to swap/modify the path to make it longer
    # Try removing a vertex and reinserting elsewhere (2-opt style for paths)
    def local_search(path):
        """Try local improvements on the path."""
        path = list(path)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            path_set = set(path)
            for i in range(len(path)):
                if time.time() > deadline:
                    return path
                for j in range(i+1, len(path)):
                    if time.time() > deadline:
                        return path
                    # Try removing path[j] and inserting at position i
                    if i == 0:
                        # Insert at beginning
                        removed = path[j]
                        new_path = path[:j] + path[j+1:]
                        # Check if removed can connect to new path[0]
                        if removed in adj_list[new_path[0]]:
                            candidate = [removed] + new_path
                            if len(candidate) == len(path):  # Same length, skip
                                continue
                            # Actually removing doesn't change length, so skip this
                            pass
                    # Try swapping path[i] and path[j] if it maintains validity
                    # This is complex, so let's try a simpler approach:
                    # Try to find a vertex not in path that can be inserted
                    pass
            
            # Try inserting a missing vertex
            for v in range(n):
                if time.time() > deadline:
                    return path
                if v in path_set:
                    continue
                # Try inserting v between path[i] and path[i+1]
                for i in range(len(path) - 1):
                    if time.time() > deadline:
                        return path
                    if v in adj_list[path[i]] and v in adj_list[path[i+1]]:
                        new_path = path[:i+1] + [v] + path[i+1:]
                        if len(new_path) > len(path):
                            path = new_path
                            path_set = set(path)
                            improved = True
                            break
                if improved:
                    break
        return path
    
    best_path = local_search(best_path)
    
    # Strategy 5: Try to find a Hamiltonian path or near-Hamiltonian path
    # Using a more systematic DFS with pruning
    def systematic_dfs(start):
        """Systematic DFS with better pruning."""
        nonlocal best_path
        if time.time() > deadline:
            return
        path = [start]
        seen = {start}
        
        def dfs(current):
            nonlocal best_path
            if time.time() > deadline:
                return
            if len(path) > len(best_path):
                best_path = list(path)
            candidates = [v for v in adj_list[current] if v not in seen]
            if not candidates:
                return
            # Sort by degree
            candidates.sort(key=lambda w: -degree[w])
            for v in candidates:
                if time.time() > deadline:
                    return
                path.append(v)
                seen.add(v)
                dfs(v)
                path.pop()
                seen.remove(v)
        
        dfs(start)
    
    # Run systematic DFS from a few starts if time permits
    if time.time() < deadline - 30:  # Only if we have at least 30 seconds left
        for start in promising_starts[:5]:
            if time.time() > deadline - 5:
                break
            systematic_dfs(start)
    
    # Final extension attempt
    best_path = try_extend(best_path)
    
    # Ensure the result is valid
    if len(best_path) == 0:
        best_path = [0]
    
    return best_path

# EVOLVE-BLOCK-END
