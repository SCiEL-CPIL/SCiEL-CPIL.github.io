import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Find Max Clique using Bron-Kerbosch
    def bronk(R, P_set, X):
        if not P_set and not X:
            return R
        max_clique = []
        # Pivot selection
        if P_set or X:
            pivot = max(list(P_set | X), key=lambda v: len(adj[v] & P_set))
        else:
            return R
        candidates = P_set - adj[pivot]
        for v in list(candidates):
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            result = bronk(new_R, new_P, new_X)
            if len(result) > len(max_clique):
                max_clique = result
            P_set.remove(v)
            X.add(v)
        return max_clique

    max_clique = bronk(set(), set(range(n)), set())
    
    def verify_and_construct(branches):
        """
        Given a set of branches, try to find disjoint paths.
        Returns dict if successful, None otherwise.
        """
        b_set = set(branches)
        # Pairs to connect
        pairs = list(combinations(sorted(branches), 2))
        
        # Graph without branches
        # We need paths in G \ B
        # Path from u to v goes u -> ... -> v
        # Interior is in G \ B
        
        # For each pair (u, v), find all simple paths of length <= 4 in G \ B
        # that start with a neighbor of u and end with a neighbor of v.
        # Actually, the path is u, x1, ..., xk, v.
        # x1 in adj[u] \ B, xk in adj[v] \ B.
        # x1...xk are in G \ B and distinct.
        
        possible_paths = {}
        for u, v in pairs:
            paths = []
            # BFS/DFS from neighbors of u
            start_nodes = [x for x in adj[u] if x not in b_set]
            for start in start_nodes:
                # DFS
                stack = [(start, [start])]
                while stack:
                    node, path = stack.pop()
                    if node == v: # Should not happen as v is in B
                        continue
                    # Check if we can reach v
                    # If node is a neighbor of v, we can finish
                    if v in adj[node]:
                        # Path is u, path..., node, v
                        # Interior is path
                        # Check if interior is valid (no branches, distinct)
                        # path contains no branches by construction
                        # Check distinctness
                        if len(set(path)) == len(path):
                            full_path = [u] + path + [v]
                            paths.append(full_path)
                    if len(path) < 4: # Limit depth
                        for nei in adj[node]:
                            if nei not in b_set and nei not in path:
                                stack.append((nei, path + [nei]))
            possible_paths[(u, v)] = paths
        
        # Now we need to select one path from each list such that interiors are disjoint.
        # Backtracking
        def backtrack(idx, used_interiors, current_paths):
            if idx == len(pairs):
                return current_paths
            u, v = pairs[idx]
            # Sort candidates by length (shorter is better for disjointness?)
            # Or by number of vertices used?
            candidates = possible_paths[(u, v)]
            # Heuristic: try paths with fewer vertices first
            candidates.sort(key=len)
            for p in candidates:
                interior = set(p[1:-1])
                if not interior.intersection(used_interiors):
                    new_used = used_interiors.union(interior)
                    res = backtrack(idx + 1, new_used, current_paths + [p])
                    if res:
                        return res
            return None

        # Order pairs by number of options (most constrained first)
        pairs_sorted = sorted(pairs, key=lambda p: len(possible_paths[p]))
        # Re-index possible_paths for sorted pairs?
        # Just use the sorted pairs in backtrack
        
        # Modify backtrack to use sorted pairs
        def backtrack_sorted(idx, used_interiors, current_paths):
            if idx == len(pairs_sorted):
                return current_paths
            u, v = pairs_sorted[idx]
            candidates = possible_paths[(u, v)]
            candidates.sort(key=len)
            for p in candidates:
                interior = set(p[1:-1])
                if not interior.intersection(used_interiors):
                    new_used = used_interiors.union(interior)
                    res = backtrack_sorted(idx + 1, new_used, current_paths + [p])
                    if res:
                        return res
            return None

        paths = backtrack_sorted(0, set(), [])
        if paths:
            # Map back to original pair order? 
            # The return format just needs a list of paths.
            # The keys 'branches' and 'paths' are required.
            # 'paths' is a list of lists.
            # The order of paths in the list doesn't matter, as long as they correspond to the pairs.
            # The problem says "paths: one simple path for each unordered pair".
            # It doesn't specify an order.
            return {
                'branches': list(branches),
                'paths': [list(p) for p in paths]
            }
        return None

    # 2. Try to expand the max clique
    best_result = verify_and_construct(max_clique)
    if not best_result:
        # Fallback to a simple edge
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    best_size = len(best_result['branches'])
    
    # Candidate vertices to add
    candidates = [v for v in range(n) if v not in max_clique]
    # Sort by degree
    candidates.sort(key=lambda v: len(adj[v]), reverse=True)
    
    start_time = time.time()
    timeout = 85.0 # Leave 5s margin
    
    # Try adding 1 vertex
    for v in candidates:
        if time.time() - start_time > timeout:
            break
        new_branches = list(max_clique) + [v]
        res = verify_and_construct(new_branches)
        if res and len(res['branches']) > best_size:
            best_result = res
            best_size = len(res['branches'])
            # If we found a larger one, we might want to try expanding further from this one?
            # But for now, just update best.
            
    # Try adding 2 vertices
    # This is expensive. Only try if best_size is small or time allows.
    if best_size < 8 and time.time() - start_time < timeout / 2:
        for v1 in candidates[:10]:
            if time.time() - start_time > timeout:
                break
            for v2 in candidates[:10]:
                if v1 == v2:
                    continue
                new_branches = list(max_clique) + [v1, v2]
                # Check if v1, v2 are distinct and not in max_clique
                if v1 in max_clique or v2 in max_clique:
                    continue
                res = verify_and_construct(new_branches)
                if res and len(res['branches']) > best_size:
                    best_result = res
                    best_size = len(res['branches'])
                    if time.time() - start_time > timeout:
                        break

    # Ensure types are correct
    final_branches = [int(b) for b in best_result['branches']]
    final_paths = [[int(x) for x in p] for p in best_result['paths']]
    
    return {'branches': final_branches, 'paths': final_paths}
# EVOLVE-BLOCK-END
