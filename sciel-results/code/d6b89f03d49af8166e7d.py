"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def are_adjacent(u, v):
        return v in adj[u]
    
    def find_shortest_path(s, t, forbidden):
        """Find shortest path from s to t avoiding forbidden vertices (except s and t)."""
        if s == t:
            return [s]
        if s in forbidden and s != t:
            return None
        if t in forbidden and t != s:
            return None
        
        # BFS
        parent = {}
        visited = {s}
        queue = deque([s])
        
        while queue:
            curr = queue.popleft()
            if curr == t:
                # Reconstruct path
                path = [t]
                while path[-1] != s:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            
            for neighbor in adj[curr]:
                if neighbor not in visited and (neighbor not in forbidden or neighbor == t):
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)
        
        return None
    
    def find_paths_for_clique(clique):
        """
        Given a clique (list of branch vertices), try to find vertex-disjoint paths
        between all pairs of branch vertices.
        Returns (paths_dict, success) where paths_dict maps (u,v) to path.
        """
        k = len(clique)
        if k < 2:
            return {}, False
        
        # We need to find paths for all C(k,2) pairs
        # Strategy: use backtracking with constraint propagation
        # Order pairs by "difficulty" (shortest path length) to fail early
        
        pairs = list(combinations(range(k), 2))
        
        # For each pair, compute shortest path length (ignoring other constraints)
        pair_info = []
        for i, j in pairs:
            s, t = clique[i], clique[j]
            # Shortest path without any forbidden vertices
            path = find_shortest_path(s, t, set())
            if path is None:
                return {}, False
            pair_info.append((len(path), i, j, s, t))
        
        # Sort by path length (easiest first, or hardest first?)
        # Let's try hardest first (longest shortest path) to constrain early
        pair_info.sort(key=lambda x: -x[0])
        
        best_paths = {}
        best_success = False
        
        def backtrack(idx, used_interior, current_paths):
            nonlocal best_paths, best_success
            
            if idx == len(pairs):
                # All paths found
                best_paths = dict(current_paths)
                best_success = True
                return True
            
            _, i, j, s, t = pair_info[idx]
            
            # Try to find a path from s to t avoiding used_interior
            # We need to search for paths, not just shortest
            # Use DFS/BFS to find all simple paths up to some length limit
            
            # First try the shortest path
            forbidden = set(used_interior)
            # s and t are allowed even if in forbidden (but they shouldn't be since they're branches)
            
            # Find all simple paths from s to t avoiding forbidden, up to reasonable length
            # Use DFS with pruning
            
            found_paths = []
            max_len = n  # upper bound on path length
            
            def dfs_paths(curr, path, visited_local):
                if len(found_paths) > 20:  # Limit number of paths to try
                    return
                if curr == t:
                    if len(path) >= 2:  # At least s and t
                        found_paths.append(list(path))
                    return
                if len(path) > max_len:
                    return
                
                for neighbor in adj[curr]:
                    if neighbor == s:
                        continue  # Don't revisit start
                    if neighbor in visited_local:
                        continue
                    if neighbor in forbidden and neighbor != t:
                        continue
                    
                    visited_local.add(neighbor)
                    path.append(neighbor)
                    dfs_paths(neighbor, path, visited_local)
                    path.pop()
                    visited_local.remove(neighbor)
            
            dfs_paths(s, [s], {s})
            
            # Sort found paths by length (prefer shorter)
            found_paths.sort(key=lambda p: len(p))
            
            for path in found_paths:
                # Check if interior vertices are disjoint from used_interior
                interior = set(path[1:-1])
                if interior & used_interior:
                    continue
                
                # Also check interior doesn't include any branch vertex
                if any(b in interior for b in clique):
                    continue
                
                new_used = used_interior | interior
                current_paths[(i, j)] = path
                
                if backtrack(idx + 1, new_used, current_paths):
                    return True
                else:
                    del current_paths[(i, j)]
            
            return False
        
        backtrack(0, set(), {})
        
        if best_success:
            return best_paths, True
        else:
            return {}, False
    
    def bron_kerbosch_max_clique():
        """Find maximum clique using Bron-Kerbosch with pivot."""
        # Convert to set representation for speed
        adj_sets = [set(adj[i]) for i in range(n)]
        
        max_clique = []
        
        def bk(R, P, X):
            nonlocal max_clique
            
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
                return
            
            # Choose pivot u from P ∪ X with maximum degree in P
            if P:
                pivot_candidates = list(P)
            elif X:
                pivot_candidates = list(X)
            else:
                return
            
            if not pivot_candidates:
                return
            
            # Pick pivot with max |P ∩ N(u)|
            pivot = max(pivot_candidates, key=lambda u: len(P & adj_sets[u]))
            
            # Recurse on P \\ N(pivot)
            candidates = list(P - adj_sets[pivot])
            
            for v in candidates:
                if time.time() > deadline:
                    return
                
                new_R = R | {v}
                new_P = P & adj_sets[v]
                new_X = X & adj_sets[v]
                
                # Pruning: if |R| + |new_P| <= len(max_clique), skip
                if len(new_R) + len(new_P) <= len(max_clique):
                    continue
                
                bk(new_R, new_P, new_X)
                
                P.remove(v)
                X.add(v)
        
        initial_P = set(range(n))
        bk(set(), initial_P, set())
        
        return max_clique
    
    # Deadline guard
    deadline = time.time() + 85.0
    
    # Strategy:
    # 1. Find a large clique in the graph
    # 2. Try to find vertex-disjoint paths for that clique
    # 3. If fails, try smaller cliques
    
    # First, find maximum clique
    max_clique = bron_kerbosch_max_clique()
    
    # Try to find paths for the max clique
    if len(max_clique) >= 2:
        paths, success = find_paths_for_clique(max_clique)
        if success:
            # Format output
            branches = list(max_clique)
            path_list = []
            for (i, j) in combinations(range(len(branches)), 2):
                path = paths[(i, j)]
                # Ensure path starts and ends at the correct branches
                path_list.append(path)
            
            return {'branches': branches, 'paths': path_list}
    
    # If max clique didn't work, try all cliques of decreasing size
    # This is expensive, so limit it
    
    # Alternative: use a different approach
    # Try to find a set of branches and paths directly using a different method
    
    # Let's try: pick a vertex, find its neighborhood, work within that
    # Or use a greedy approach
    
    # Greedy: start with empty set, add vertices one by one if we can maintain path disjointness
    # This is complex, so let's try a simpler heuristic
    
    # Try all pairs of vertices and extend
    best_result = None
    
    # Try cliques of size up to min(len(max_clique), 8)
    max_size_to_try = min(len(max_clique) + 1, 10)
    
    for size in range(max_size_to_try, 1, -1):
        if time.time() > deadline - 5.0:
            break
        
        if size > n:
            continue
        
        # Find all cliques of this size (limit to avoid explosion)
        # Use a simple approach: for each vertex, try to extend
        found_any = False
        
        # Use a different strategy: sample random subsets
        import random
        random.seed(42)
        
        attempts = 0
        max_attempts = 5000 if size <= 6 else 500
        
        while attempts < max_attempts and time.time() < deadline - 3.0:
            attempts += 1
            
            # Random subset of size 'size'
            subset = random.sample(range(n), size)
            
            # Check if it's a clique
            is_clique = True
            for i in range(size):
                for j in range(i+1, size):
                    if not are_adjacent(subset[i], subset[j]):
                        is_clique = False
                        break
                if not is_clique:
                    break
            
            if not is_clique:
                continue
            
            # Try to find paths
            paths, success = find_paths_for_clique(subset)
            if success:
                branches = list(subset)
                path_list = []
                for (i, j) in combinations(range(len(branches)), 2):
                    path = paths[(i, j)]
                    path_list.append(path)
                
                if best_result is None or len(branches) > len(best_result['branches']):
                    best_result = {'branches': branches, 'paths': path_list}
                    found_any = True
                
                if len(branches) >= size:
                    # Found a clique of this size that works
                    # We could try to find larger ones, but for now return
                    pass
        
        if found_any and best_result is not None:
            # Found a valid result of this size, try larger sizes already handled
            # Since we're going from large to small, if we found one, we can return
            # But we want the maximum, so continue trying larger... wait, we go from large to small
            # So if we found one at this size, we should return since we're trying larger first
            return best_result
    
    # Fallback: return a trivial result
    if best_result is not None:
        return best_result
    
    # Very basic fallback: find any edge
    if edges:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Should not reach here
    return {'branches': [0, 1], 'paths': [[0, 1]]}
# EVOLVE-BLOCK-END
