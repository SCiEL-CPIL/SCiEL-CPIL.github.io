import time
from collections import deque
from itertools import combinations
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]

    def time_ok():
        return time.time() < deadline

    # Max-flow based disjoint paths finder
    def max_flow_disjoint_paths(branches, max_path_len=6):
        k = len(branches)
        if k < 2:
            return None
        
        branches_set = set(branches)
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((branches[i], branches[j]))
        
        # We need to find internally disjoint paths for all pairs.
        # This is a complex constraint satisfaction problem.
        # A greedy approach with max-flow for each pair sequentially often fails
        # because early choices block later ones.
        
        # Strategy:
        # 1. Use a backtracking search over pairs.
        # 2. For each pair, find a set of candidate paths (shortest first).
        # 3. Try each candidate path, checking for disjointness with previously chosen paths.
        # 4. To speed up candidate generation, use BFS to find shortest paths avoiding used vertices.
        
        # Optimization: Order pairs by "difficulty" (fewest available paths) dynamically.
        
        used_interior = set()
        
        def find_shortest_paths(s, t, forbidden, limit=10):
            # BFS to find shortest paths
            # Returns list of paths
            if s in forbidden or t in forbidden:
                return []
            if s == t:
                return [[s]]
            
            # BFS for shortest path
            # We want a set of diverse paths, not just one.
            # Let's do a DFS with length limit to find multiple short paths.
            paths = []
            
            def dfs(node, path, visited):
                if not time_ok():
                    return
                if node == t:
                    paths.append(list(path))
                    return
                
                if len(path) - 1 >= limit:
                    return
                
                for nb in adj_list[node]:
                    if nb in visited:
                        continue
                    if nb in forbidden and nb != t:
                        continue
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.remove(nb)
            
            visited = {s}
            path = [s]
            dfs(s, path, visited)
            
            # Sort by length
            paths.sort(key=len)
            return paths

        def backtrack(remaining_pairs, used_interior, path_map):
            if not time_ok():
                return None
            
            if not remaining_pairs:
                return path_map
            
            # MRV Heuristic: Pick pair with fewest available paths
            min_paths_count = float('inf')
            best_pair_idx = -1
            candidate_paths = []
            
            for i, (u, v) in enumerate(remaining_pairs):
                forbidden = branches_set - {u, v}
                forbidden_with_used = forbidden | used_interior
                paths = find_shortest_paths(u, v, forbidden_with_used, limit=6)
                if len(paths) < min_paths_count:
                    min_paths_count = len(paths)
                    best_pair_idx = i
                    candidate_paths = paths
                if min_paths_count == 0:
                    break
            
            if best_pair_idx == -1 or min_paths_count == 0:
                return None
            
            u, v = remaining_pairs[best_pair_idx]
            new_remaining = [p for i, p in enumerate(remaining_pairs) if i != best_pair_idx]
            
            # Try each candidate path
            # Heuristic: Prefer shorter paths
            for p in candidate_paths:
                interior = set(p[1:-1])
                if interior & used_interior:
                    continue
                
                new_path_map = dict(path_map)
                new_path_map[(u, v)] = p
                
                result = backtrack(new_remaining, used_interior | interior, new_path_map)
                if result is not None:
                    return result
            
            return None

        result = backtrack(pairs, set(), {})
        return result

    # Find max clique using Bron-Kerbosch with pivot
    def find_max_clique():
        best = []
        
        def bk(R, Pset, X):
            nonlocal best
            if not time_ok():
                return
            if not Pset and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            
            if Pset:
                pivot_candidates = list(Pset | X)
            else:
                pivot_candidates = list(X)
            
            if not pivot_candidates:
                return
            
            u = max(pivot_candidates, key=lambda v: len(adj[v] & Pset))
            candidates = [v for v in Pset if v not in adj[u]]
            candidates.sort(key=lambda x: -len(adj[x]))
            
            for v in candidates:
                if not time_ok():
                    return
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset = Pset - {v}
                X = X | {v}
        
        bk(frozenset(), frozenset(range(n)), frozenset())
        return best

    max_clique = find_max_clique()
    best_result = None
    best_k = 0

    # Try the max clique first
    if len(max_clique) >= 2:
        for mlen in [4, 5, 6]:
            if not time_ok():
                break
            paths_dict = max_flow_disjoint_paths(max_clique, mlen)
            if paths_dict:
                best_result = (max_clique, paths_dict)
                best_k = len(max_clique)
                break

    # If full clique failed, try subsets or use local search
    if best_result is None or best_k < len(max_clique):
        # Start with a small valid set
        init_branches = None
        # Try to find a valid set of size 3 or 4 from the clique
        for size in range(min(len(max_clique), 4), 1, -1):
            if not time_ok():
                break
            for subset in combinations(max_clique, size):
                subset_list = list(subset)
                paths_dict = max_flow_disjoint_paths(subset_list, 5)
                if paths_dict:
                    init_branches = subset_list
                    break
            if init_branches:
                break
        
        if init_branches is None:
            # Fallback to an edge
            for u, v in edges:
                if not time_ok():
                    break
                paths_dict = max_flow_disjoint_paths([u, v], 3)
                if paths_dict:
                    init_branches = [u, v]
                    break
        
        if init_branches is None:
            init_branches = [edges[0][0], edges[0][1]]
            paths_dict = max_flow_disjoint_paths(init_branches, 3)
            if paths_dict is None:
                return {'branches': [0, 1], 'paths': [[0, 1]]}
            
        best_result = (init_branches, paths_dict)
        best_k = len(init_branches)

    # Local Search: Try to add vertices
    current_branches = list(best_result[0])
    current_paths = best_result[1]
    
    improved = True
    while improved and time_ok():
        improved = False
        candidates = [v for v in range(n) if v not in current_branches]
        random.shuffle(candidates)
        
        for v in candidates:
            if not time_ok():
                break
            
            new_branches = sorted(current_branches + [v])
            mlen = 5 if len(new_branches) <= 5 else 6
            new_paths = max_flow_disjoint_paths(new_branches, mlen)
            
            if new_paths:
                current_branches = new_branches
                current_paths = new_paths
                best_result = (current_branches, current_paths)
                best_k = len(current_branches)
                improved = True
                break

    # Try a few more rounds with different random order
    if time_ok():
        for _ in range(5):
            if not time_ok():
                break
            improved = False
            candidates = [v for v in range(n) if v not in current_branches]
            random.shuffle(candidates)
            for v in candidates:
                if not time_ok():
                    break
                new_branches = sorted(current_branches + [v])
                new_paths = max_flow_disjoint_paths(new_branches, 6)
                if new_paths:
                    current_branches = new_branches
                    current_paths = new_paths
                    best_result = (current_branches, current_paths)
                    best_k = len(current_branches)
                    improved = True
                    break
            if not improved:
                break

    # Construct final output
    branches_out = [int(b) for b in current_branches]
    paths_out = []
    k = len(branches_out)
    for i in range(k):
        for j in range(i+1, k):
            u, v = branches_out[i], branches_out[j]
            key = (min(u, v), max(u, v))
            p = current_paths[key]
            if p[0] == u and p[-1] == v:
                paths_out.append([int(x) for x in p])
            elif p[0] == v and p[-1] == u:
                paths_out.append([int(x) for x in reversed(p)])
            else:
                paths_out.append([int(x) for x in p])

    return {
        'branches': branches_out,
        'paths': paths_out
    }

# Run construct to test
# print(construct())
