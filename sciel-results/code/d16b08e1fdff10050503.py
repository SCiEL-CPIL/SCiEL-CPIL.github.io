import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = P['n']
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)

    def time_check():
        return time.time() > deadline

    # Precompute all-pairs shortest paths
    dist = [[-1] * n for _ in range(n)]
    parent_dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    parent_dist[s][v] = u
                    q.append(v)

    def get_k_shortest_paths(u, v, k=10, max_len=6):
        """
        Find up to k shortest simple paths from u to v.
        Returns a list of paths.
        """
        if u == v:
            return []
        if dist[u][v] == -1:
            return []
        
        paths = []
        # Use a modified Dijkstra/BFS that keeps track of the number of paths found per node
        # Or simply BFS up to max_len and collect all, then sort by length.
        # Given small n, we can just do a DFS with depth limit and collect, then sort.
        
        def dfs(cur, target, path, visited):
            if len(paths) >= k:
                return
            if cur == target:
                paths.append(list(path))
                return
            if len(path) > max_len:
                return
            
            for nxt in adj_list[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, target, path, visited)
                    path.pop()
                    visited.remove(nxt)
        
        visited = {u}
        dfs(u, v, [u], visited)
        
        # Sort by length
        paths.sort(key=lambda p: len(p))
        return paths[:k]

    def find_paths_for_branches(branches):
        """
        Try to find disjoint internal paths for all pairs of branches.
        Uses backtracking over candidate paths.
        """
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Check basic feasibility
        for u, v in pairs:
            if dist[u][v] < 0:
                return None
        
        # Precompute candidate paths for each pair
        # We want to prioritize pairs with fewer options (most constrained)
        pair_candidates = {}
        for u, v in pairs:
            # Generate candidates
            # If adjacent, direct edge is the best candidate
            if u in adj_set[v]:
                pair_candidates[(u, v)] = [[u, v]]
            else:
                cands = get_k_shortest_paths(u, v, k=20, max_len=5)
                if not cands:
                    return None
                pair_candidates[(u, v)] = cands
        
        # Sort pairs by number of candidates (ascending)
        sorted_pairs = sorted(pairs, key=lambda p: len(pair_candidates[p]))
        
        used_internals = set()
        assignment = {}
        
        def backtrack(idx):
            if time_check():
                return False
            if idx == len(sorted_pairs):
                return True
            
            u, v = sorted_pairs[idx]
            cands = pair_candidates[(u, v)]
            
            for path in cands:
                # Check if internal vertices are free
                internal = path[1:-1]
                conflict = False
                for w in internal:
                    if w in used_internals:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Assign
                for w in internal:
                    used_internals.add(w)
                assignment[(u, v)] = path
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                for w in internal:
                    used_internals.discard(w)
                del assignment[(u, v)]
            
            return False
        
        if backtrack(0):
            result_paths = {}
            for pair in pairs:
                result_paths[pair] = assignment[pair]
            return result_paths
        
        return None

    def solve():
        """Find maximum number of branches using branch-and-bound."""
        best_result = None
        best_k = 0
        
        # Sort vertices by degree descending
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        
        current_branches = []
        
        def backtrack_clique(candidate_start_idx):
            nonlocal best_result, best_k
            
            if time_check():
                return
            
            # Update best
            if len(current_branches) > best_k:
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
                    best_result = {
                        'branches': [int(b) for b in current_branches],
                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                    }
                    best_k = len(current_branches)
            
            # Try adding more vertices
            for i in range(candidate_start_idx, n):
                if time_check():
                    return
                
                candidate = sorted_vertices[i]
                
                # Pruning
                if len(current_branches) + (n - i) <= best_k:
                    break
                
                if candidate in current_branches:
                    continue
                
                # Quick feasibility check: candidate must be connected to all current branches
                feasible = True
                for b in current_branches:
                    if dist[candidate][b] < 0:
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                # Try adding candidate
                current_branches.append(candidate)
                
                # We don't necessarily need to verify paths here to recurse,
                # but it helps to prune early if it's obviously infeasible.
                # However, find_paths is expensive. 
                # Let's just recurse and let the leaf check handle it, 
                # OR do a very quick check.
                
                backtrack_clique(i + 1)
                
                current_branches.pop()
        
        backtrack_clique(0)
        
        if best_result is not None:
            return best_result
        
        # Fallback
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    return solve()
