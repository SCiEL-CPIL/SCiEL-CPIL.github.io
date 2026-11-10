import time
import sys
from itertools import combinations
from collections import deque
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin for safety
    
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
    
    def bfs_path(src, dst, blocked):
        if src == dst:
            return [src]
        if src in blocked or dst in blocked:
            return None
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for neighbor in adj_list[node]:
                if neighbor not in parent and neighbor not in blocked:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None
    
    def hopcroft_karp(feasible_pairs):
        """Hopcroft-Karp bipartite matching. Returns dict pair_idx -> internal_vertex or None."""
        if not feasible_pairs:
            return {}
        
        all_internals = set()
        for _, candidates in feasible_pairs:
            all_internals.update(candidates)
        
        left_nodes = [p[0] for p in feasible_pairs]
        right_nodes = list(all_internals)
        
        left_to_idx = {node: i for i, node in enumerate(left_nodes)}
        right_to_idx = {node: i for i, node in enumerate(right_nodes)}
        
        adj = [[] for _ in range(len(left_nodes))]
        for pair_idx, candidates in feasible_pairs:
            li = left_to_idx[pair_idx]
            for w in candidates:
                if w in right_to_idx:
                    adj[li].append(right_to_idx[w])
        
        pair_u = [-1] * len(left_nodes)
        pair_v = [-1] * len(right_nodes)
        dist = [0] * len(left_nodes)
        
        def bfs():
            q = deque()
            found = False
            for u in range(len(left_nodes)):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching_size = 0
        while bfs():
            for u in range(len(left_nodes)):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1
        
        if matching_size != len(left_nodes):
            return None
        
        result = {}
        for u in range(len(left_nodes)):
            if pair_u[u] != -1:
                result[left_nodes[u]] = right_nodes[pair_u[u]]
        return result
    
    def find_paths_for_branches(branches):
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        direct_pairs = []
        common_pairs = []
        hard_pairs = []
        
        for u, v in pairs:
            if u in adj_set[v]:
                direct_pairs.append((u, v))
            else:
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_set[v]:
                        common.add(w)
                if common:
                    common_pairs.append((u, v, common))
                else:
                    hard_pairs.append((u, v))
        
        all_non_direct = [(u, v) for u, v, _ in common_pairs] + [(u, v) for u, v in hard_pairs]
        assignment = {}
        
        if hard_pairs:
            # Must use general BFS backtracking for all non-direct pairs
            used_internals = set()
            
            # Sort: hard pairs (0 common) first, then common pairs by count
            sorted_pairs = [(u, v, 0) for u, v in hard_pairs] + [(u, v, len(c)) for u, v, c in common_pairs]
            sorted_pairs.sort(key=lambda x: x[2])
            
            def backtrack_paths(idx):
                if time_check():
                    return False
                if idx == len(sorted_pairs):
                    return True
                u, v, _ = sorted_pairs[idx]
                blocked = branch_set | used_internals
                blocked.discard(u)
                blocked.discard(v)
                path = bfs_path(u, v, blocked)
                if path is None:
                    return False
                internal = path[1:-1]
                if len(internal) != len(set(internal)):
                    return False
                for w in internal:
                    if w in branch_set:
                        return False
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
            # No hard pairs. Try fast methods on common_pairs.
            if not common_pairs:
                pass
            else:
                # 1. Greedy
                used_internals = set()
                greedy_success = True
                sorted_common = sorted(common_pairs, key=lambda x: len(x[2]))
                for u, v, common in sorted_common:
                    candidates = [w for w in common if w not in used_internals]
                    if not candidates:
                        greedy_success = False
                        break
                    chosen = candidates[0]
                    used_internals.add(chosen)
                    assignment[(u, v)] = [chosen]
                
                if not greedy_success:
                    # 2. Hopcroft-Karp
                    assignment = {}
                    pair_tuples = [(u, v) for u, v, _ in common_pairs]
                    pair_to_idx = {p: i for i, p in enumerate(pair_tuples)}
                    feasible_list = [(pair_to_idx[(u, v)], list(c)) for u, v, c in common_pairs]
                    matching = hopcroft_karp(feasible_list)
                    if matching is None:
                        # 3. Backtracking BFS on common_pairs
                        used_internals = set()
                        assignment = {}
                        sorted_common_bfs = sorted(common_pairs, key=lambda x: len(x[2]))
                        
                        def backtrack_paths_common(idx):
                            if time_check():
                                return False
                            if idx == len(sorted_common_bfs):
                                return True
                            u, v, _ = sorted_common_bfs[idx]
                            blocked = branch_set | used_internals
                            blocked.discard(u)
                            blocked.discard(v)
                            path = bfs_path(u, v, blocked)
                            if path is None:
                                return False
                            internal = path[1:-1]
                            if len(internal) != len(set(internal)):
                                return False
                            for w in internal:
                                if w in branch_set:
                                    return False
                            for w in internal:
                                used_internals.add(w)
                            assignment[(u, v)] = internal
                            if backtrack_paths_common(idx + 1):
                                return True
                            for w in internal:
                                used_internals.discard(w)
                            del assignment[(u, v)]
                            return False
                        
                        if not backtrack_paths_common(0):
                            return None
                    else:
                        for pair_idx, w in matching.items():
                            p = pair_tuples[pair_idx]
                            assignment[p] = [w]
        
        result_paths = {}
        for u, v in direct_pairs:
            result_paths[(u, v)] = [u, v]
        for u, v in all_non_direct:
            internal = assignment.get((u, v), [])
            if not internal:
                return None
            result_paths[(u, v)] = [u] + internal + [v]
        
        return result_paths
    
    def greedy_initial():
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        branches = []
        for v in sorted_vertices:
            if time_check():
                break
            test = branches + [v]
            if find_paths_for_branches(test) is not None:
                branches.append(v)
        return branches
    
    def local_search_improve(branches):
        """Try to improve branch set by swapping out one vertex for another."""
        if not branches:
            return branches
        
        degree = [len(adj_list[i]) for i in range(n)]
        branch_set = set(branches)
        
        # Sort candidates by degree descending, excluding current branches
        candidates = [v for v in range(n) if v not in branch_set]
        candidates.sort(key=lambda x: -degree[x])
        
        best_branches = list(branches)
        best_size = len(branches)
        
        # Try removing each branch and adding the best candidate
        for i, remove_v in enumerate(list(branches)):
            if time_check():
                break
            
            # Remove branch_v
            reduced = [b for j, b in enumerate(branches) if j != i]
            
            # Try adding candidates in order
            for add_v in candidates:
                if time_check():
                    break
                
                new_branches = reduced + [add_v]
                if find_paths_for_branches(new_branches) is not None:
                    # Check if we can further improve
                    best_branches = new_branches
                    best_size = len(new_branches)
                    break
        
        # If we improved, try to add more
        if best_size > len(branches):
            # Try adding more vertices to the improved set
            branch_set = set(best_branches)
            for v in candidates:
                if time_check():
                    break
                if v in branch_set:
                    continue
                test = best_branches + [v]
                if find_paths_for_branches(test) is not None:
                    best_branches.append(v)
                    branch_set.add(v)
        
        return best_branches
    
    def solve():
        best_result = None
        best_k = 0
        
        greedy_branches = greedy_initial()
        if greedy_branches:
            paths = find_paths_for_branches(greedy_branches)
            if paths is not None:
                best_k = len(greedy_branches)
                best_result = {
                    'branches': [int(b) for b in greedy_branches],
                    'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                }
        
        # Local search improvement
        if best_result is not None and not time_check():
            improved = local_search_improve(greedy_branches)
            if len(improved) > best_k:
                paths = find_paths_for_branches(improved)
                if paths is not None:
                    best_k = len(improved)
                    best_result = {
                        'branches': [int(b) for b in improved],
                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                    }
                    greedy_branches = improved
        
        degree = [len(adj_list[i]) for i in range(n)]
        sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
        
        current_branches = []
        
        def backtrack_clique(candidate_start_idx):
            nonlocal best_result, best_k
            
            if time_check():
                return
            
            if len(current_branches) > best_k:
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
                    best_result = {
                        'branches': [int(b) for b in current_branches],
                        'paths': [[int(x) for x in paths[p]] for p in sorted(paths.keys())]
                    }
                    best_k = len(current_branches)
            
            for i in range(candidate_start_idx, n):
                if time_check():
                    return
                
                candidate = sorted_vertices[i]
                remaining = n - i
                if len(current_branches) + remaining <= best_k:
                    break
                
                if candidate in current_branches:
                    continue
                
                current_branches.append(candidate)
                paths = find_paths_for_branches(current_branches)
                if paths is not None:
                    backtrack_clique(i + 1)
                current_branches.pop()
        
        backtrack_clique(0)
        
        if best_result is not None:
            return best_result
        
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    return solve()
# EVOLVE-BLOCK-END
