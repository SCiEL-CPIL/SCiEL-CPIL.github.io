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

    # Precompute all-pairs shortest paths (BFS) for quick feasibility checks
    # dist[u][v] = shortest path length, -1 if no path
    dist = [[-1] * n for _ in range(n)]
    parent_dist = [[-1] * n for _ in range(n)]  # parent_dist[u][v] = next hop from u to v
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

    def find_paths_for_branches(branches):
        """Try to find disjoint internal paths for all pairs of branches.
        Returns dict of (u,v) -> path, or None if infeasible."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Check basic feasibility: every pair must be connected
        for u, v in pairs:
            if dist[u][v] < 0:
                return None
        
        # For each non-adjacent pair, find candidate internal vertices
        # A vertex w can be internal for pair (u,v) if:
        #   - w is not a branch
        #   - there exists a path u -> ... -> w -> ... -> v using only non-branch, non-used vertices
        # Simplification: try length-2 paths first (w adjacent to both u and v)
        
        pair_info = {}
        non_direct_pairs = []
        for u, v in pairs:
            if u in adj_set[v]:
                pair_info[(u, v)] = ('direct', None)
            else:
                # Find internal vertices adjacent to both
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_set[v]:
                        common.add(w)
                if not common:
                    # Try longer paths - find any vertex on a shortest path
                    # that is not a branch
                    path_len = dist[u][v]
                    if path_len < 2:
                        return None
                    # Get the shortest path
                    path = [v]
                    cur = v
                    while cur != u:
                        cur = parent_dist[u][cur]
                        path.append(cur)
                    path.reverse()
                    # All internal vertices on this path are candidates
                    common = set(path[1:-1]) - branch_set
                    if not common:
                        return None
                pair_info[(u, v)] = ('need', common)
                non_direct_pairs.append((u, v))
        
        # Try to assign distinct internal vertices
        if non_direct_pairs:
            # Sort by fewest options first
            sorted_pairs = sorted(non_direct_pairs, key=lambda p: len(pair_info[p][1]))
            
            assignment = {}
            used = set()
            
            def backtrack_match(idx):
                if time_check():
                    return False
                if idx == len(sorted_pairs):
                    return True
                pair = sorted_pairs[idx]
                for w in pair_info[pair][1]:
                    if w not in used:
                        used.add(w)
                        assignment[pair] = [w]
                        if backtrack_match(idx + 1):
                            return True
                        used.remove(w)
                        del assignment[pair]
                return False
            
            if not backtrack_match(0):
                # Try with BFS-based longer paths
                assignment = {}
                used_internals = set()
                
                def backtrack_paths(idx):
                    if time_check():
                        return False
                    if idx == len(sorted_pairs):
                        return True
                    u, v = sorted_pairs[idx]
                    
                    blocked = branch_set | used_internals
                    blocked.discard(u)
                    blocked.discard(v)
                    
                    # BFS shortest path avoiding blocked
                    parent = {u: -1}
                    queue = deque([u])
                    found = False
                    while queue and not found:
                        node = queue.popleft()
                        for neighbor in adj_list[node]:
                            if neighbor not in parent and neighbor not in blocked:
                                parent[neighbor] = node
                                if neighbor == v:
                                    found = True
                                    break
                                queue.append(neighbor)
                    
                    if not found:
                        return False
                    
                    path = []
                    cur = v
                    while cur != -1:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    
                    internal = path[1:-1]
                    
                    for w in internal:
                        used_internals.add(w)
                    assignment[(u, v)] = internal
                    
                    if backtrack_paths(idx + 1):
                        return True
                    
                    for w in internal:
                        used_internals.discard(w)
                    del assignment[(u, v)]
                    
                    return False
                
                if not backtrack_paths(0):
                    return None
        else:
            assignment = {}
        
        result_paths = {}
        for pair in pairs:
            u, v = pair
            if pair_info[pair][0] == 'direct':
                result_paths[pair] = [u, v]
            else:
                internal = assignment[pair]
                result_paths[pair] = [u] + internal + [v]
        
        return result_paths

    def solve():
        """Find maximum number of branches using branch-and-bound."""
        best_result = None
        best_k = 0
        
        # Sort vertices by degree descending
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        
        # Precompute compatibility: can vertex v be added to a set S?
        # Quick check: v must be connected to all in S with enough internal vertices available
        
        current_branches = []
        
        def upper_bound(current_size, start_idx):
            """Upper bound on how many more vertices we can add."""
            # Simple bound: remaining vertices
            return n - start_idx
        
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
                
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
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
