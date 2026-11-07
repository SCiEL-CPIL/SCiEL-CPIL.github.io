import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    def time_up():
        return time.time() - start_time > deadline

    def find_simple_paths(u, v, banned, limit=20):
        """Find up to 'limit' simple paths from u to v avoiding 'banned' vertices in interior."""
        paths = []
        if u in banned or v in banned:
            return paths
        if u == v:
            return [[u]]
            
        # DFS to find paths
        def dfs(cur, path, visited):
            if time_up():
                return
            if len(paths) >= limit:
                return
            if cur == v:
                paths.append(list(path))
                return
            
            for nb in adj[cur]:
                if nb in visited:
                    continue
                if nb in banned and nb != v:
                    continue
                    
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
                
        dfs(u, [u], {u})
        # Sort by length
        paths.sort(key=len)
        return paths

    def solve_disjoint_paths(branches):
        """Find disjoint paths for all pairs of branches."""
        if time_up():
            return None
            
        pairs = list(combinations(branches, 2))
        if not pairs:
            return []
            
        # Order pairs by number of candidate paths (fewest first)
        # This is a heuristic to find a solution faster
        pair_candidates = {}
        for u, v in pairs:
            banned = set(branches) - {u, v}
            # For the initial candidate generation, we don't ban any interiors yet
            # because we don't know which ones will be used.
            # But we can ban the other branches.
            candidates = find_simple_paths(u, v, banned, limit=10)
            if not candidates:
                return None # No path exists for this pair
            pair_candidates[(u, v)] = candidates
            
        # Sort pairs by number of candidates
        pairs_sorted = sorted(pairs, key=lambda p: len(pair_candidates[p]))
        
        # Backtracking search
        assigned_paths = {}
        used_interiors = set()
        
        def backtrack(idx):
            if time_up():
                return False
            if idx == len(pairs_sorted):
                return True
                
            u, v = pairs_sorted[idx]
            # Filter candidates that don't use used_interiors
            valid_candidates = []
            for path in pair_candidates[(u, v)]:
                interior = set(path[1:-1])
                if not interior.issubset(used_interiors): # Check if any interior is used
                    # Actually, we need to check if interior is disjoint from used_interiors
                    if not interior & used_interiors:
                        valid_candidates.append(path)
                        
            # Try each valid candidate
            for path in valid_candidates:
                interior = set(path[1:-1])
                used_interiors.update(interior)
                assigned_paths[(u, v)] = path
                
                if backtrack(idx + 1):
                    return True
                    
                used_interiors.difference_update(interior)
                del assigned_paths[(u, v)]
                
            return False
            
        if backtrack(0):
            # Return paths in a consistent order
            result = []
            for u, v in pairs:
                result.append(assigned_paths[(u, v)])
            return result
        return None

    # 1. Find all cliques of size 3 to 6
    best_solution = None
    best_size = 0
    
    # Find K3
    for combo in combinations(range(n), 3):
        if time_up():
            break
        is_clique = True
        for u, v in combinations(combo, 2):
            if (u, v) not in edge_set:
                is_clique = False
                break
        if is_clique:
            branches = list(combo)
            paths = solve_disjoint_paths(branches)
            if paths:
                if len(branches) > best_size:
                    best_size = len(branches)
                    best_solution = {'branches': branches, 'paths': paths}
                    
    # Find K4
    for combo in combinations(range(n), 4):
        if time_up():
            break
        is_clique = True
        for u, v in combinations(combo, 2):
            if (u, v) not in edge_set:
                is_clique = False
                break
        if is_clique:
            branches = list(combo)
            paths = solve_disjoint_paths(branches)
            if paths:
                if len(branches) > best_size:
                    best_size = len(branches)
                    best_solution = {'branches': branches, 'paths': paths}
                    
    # Find K5
    for combo in combinations(range(n), 5):
        if time_up():
            break
        is_clique = True
        for u, v in combinations(combo, 2):
            if (u, v) not in edge_set:
                is_clique = False
                break
        if is_clique:
            branches = list(combo)
            paths = solve_disjoint_paths(branches)
            if paths:
                if len(branches) > best_size:
                    best_size = len(branches)
                    best_solution = {'branches': branches, 'paths': paths}
                    
    # Find K6
    for combo in combinations(range(n), 6):
        if time_up():
            break
        is_clique = True
        for u, v in combinations(combo, 2):
            if (u, v) not in edge_set:
                is_clique = False
                break
        if is_clique:
            branches = list(combo)
            paths = solve_disjoint_paths(branches)
            if paths:
                if len(branches) > best_size:
                    best_size = len(branches)
                    best_solution = {'branches': branches, 'paths': paths}
                    
    # If no clique of size >= 4 was found, try to expand a K4 by adding a vertex
    # This is a heuristic for non-clique branch sets
    if best_size < 4:
        # Find a K4
        k4 = None
        for combo in combinations(range(n), 4):
            if time_up():
                break
            is_clique = True
            for u, v in combinations(combo, 2):
                if (u, v) not in edge_set:
                    is_clique = False
                    break
            if is_clique:
                k4 = list(combo)
                break
                
        if k4:
            for v in range(n):
                if time_up():
                    break
                if v in k4:
                    continue
                branches = k4 + [v]
                paths = solve_disjoint_paths(branches)
                if paths:
                    if len(branches) > best_size:
                        best_size = len(branches)
                        best_solution = {'branches': branches, 'paths': paths}
                        
    if best_solution:
        return best_solution
        
    # Fallback to K3
    for combo in combinations(range(n), 3):
        if time_up():
            break
        is_clique = True
        for u, v in combinations(combo, 2):
            if (u, v) not in edge_set:
                is_clique = False
                break
        if is_clique:
            branches = list(combo)
            paths = [[u, v] for u, v in combinations(branches, 2)]
            return {'branches': branches, 'paths': paths}
        
    # Fallback to edge
    u, v = edges[0]
    return {'branches': [u, v], 'paths': [[u, v]]}
