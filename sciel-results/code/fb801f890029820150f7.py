import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for 90s limit

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

    def bfs_path(u, v, blocked):
        """Find shortest path from u to v avoiding blocked vertices (except u,v themselves)."""
        if u == v:
            return [u]
        parent = {u: -1}
        queue = deque([u])
        found = False
        while queue and not found:
            node = queue.popleft()
            for neighbor in adj_list[node]:
                if neighbor not in parent:
                    if neighbor == v or neighbor not in blocked:
                        parent[neighbor] = node
                        if neighbor == v:
                            found = True
                            break
                        queue.append(neighbor)
        if not found:
            return None
        path = []
        cur = v
        while cur != -1:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def find_all_paths(branches):
        """Find vertex-disjoint internal paths for all pairs of branches.
        Uses greedy with backtracking. Returns dict of (u,v)->path or None."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        if not pairs:
            return {}
        
        # For each pair, determine if direct edge exists
        pair_data = {}
        for u, v in pairs:
            if u in adj_set[v]:
                pair_data[(u, v)] = ('direct', None)
            else:
                pair_data[(u, v)] = ('path', None)
        
        # Sort non-direct pairs by difficulty (fewest options first)
        non_direct = [(u, v) for u, v in pairs if pair_data[(u, v)][0] == 'path']
        
        # Greedy assignment with backtracking
        assignment = {}
        used_internals = set()
        
        def try_assign(idx, sorted_pairs):
            if time_check():
                return False
            if idx == len(sorted_pairs):
                return True
            u, v = sorted_pairs[idx]
            
            # Find shortest path avoiding used_internals and other branches (except u,v)
            blocked = (branch_set - {u, v}) | used_internals
            path = bfs_path(u, v, blocked)
            if path is None:
                return False
            
            internal = path[1:-1]
            
            # Try this path
            for w in internal:
                used_internals.add(w)
            assignment[(u, v)] = internal
            
            if try_assign(idx + 1, sorted_pairs):
                return True
            
            for w in internal:
                used_internals.discard(w)
            del assignment[(u, v)]
            
            # Try alternative: if there are multiple shortest paths, we'd need to explore them
            # For now, just fail (BFS gives one path; we could enhance to find all)
            return False
        
        # Sort non-direct pairs: prioritize those with fewer common neighbors
        def pair_difficulty(pair):
            u, v = pair
            common = 0
            for w in adj_list[u]:
                if w not in branch_set and w in adj_set[v]:
                    common += 1
            return common
        
        sorted_non_direct = sorted(non_direct, key=pair_difficulty)
        
        if sorted_non_direct:
            if not try_assign(0, sorted_non_direct):
                return None
        
        # Build result
        result = {}
        for u, v in pairs:
            if pair_data[(u, v)][0] == 'direct':
                result[(u, v)] = [u, v]
            else:
                result[(u, v)] = [u] + assignment[(u, v)] + [v]
        
        return result

    def solve():
        """Find maximum branch set using branch-and-bound."""
        best_result = None
        best_k = 0
        
        # Sort vertices by degree descending
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        
        current_branches = []
        best_paths = None
        
        def check_and_update():
            nonlocal best_result, best_k, best_paths
            if len(current_branches) <= best_k:
                return
            paths = find_all_paths(current_branches)
            if paths is not None:
                best_k = len(current_branches)
                best_paths = paths
                best_result = {
                    'branches': [int(b) for b in current_branches],
                    'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                }
        
        def backtrack(start_idx):
            nonlocal best_result, best_k, best_paths
            
            if time_check():
                return
            
            check_and_update()
            
            for i in range(start_idx, n):
                if time_check():
                    return
                
                # Pruning
                remaining = n - i
                if len(current_branches) + remaining <= best_k:
                    break
                
                candidate = sorted_vertices[i]
                
                # Quick feasibility check: can candidate connect to all current branches?
                feasible = True
                for b in current_branches:
                    if b not in adj_set[candidate]:
                        # Check if path exists avoiding other branches
                        blocked = set(current_branches) - {b}
                        path = bfs_path(candidate, b, blocked)
                        if path is None:
                            feasible = False
                            break
                if not feasible:
                    continue
                
                current_branches.append(candidate)
                backtrack(i + 1)
                current_branches.pop()
        
        backtrack(0)
        
        if best_result is not None:
            return best_result
        
        # Fallback: find at least 2 connected vertices
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    return solve()
