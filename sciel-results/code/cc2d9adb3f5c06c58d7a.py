"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque, defaultdict

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe (90s wall clock)
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def max_clique_bronk():
        """Bron-Kerbosch with pivoting to find maximum clique"""
        max_clique = []
        
        def bronk(R, P_set, X_set):
            nonlocal max_clique
            if time.time() > deadline:
                return
            
            if not P_set and not X_set:
                if len(R) > len(max_clique):
                    max_clique = list(R)
                return
            
            U = P_set.union(X_set)
            if not U:
                return
                
            # Find pivot with max intersection with P_set
            pivot = None
            max_intersect = -1
            for u in U:
                inter = len(P_set.intersection(adj[u]))
                if inter > max_intersect:
                    max_intersect = inter
                    pivot = u
                    
            candidates = list(P_set.difference(adj[pivot]))
            
            for v in candidates:
                bronk(R.union({v}), P_set.intersection(adj[v]), X_set.intersection(adj[v]))
                P_set.remove(v)
                X_set.add(v)
                if time.time() > deadline:
                    return

        all_vertices = set(range(n))
        bronk(set(), all_vertices, set())
        return max_clique

    def find_disjoint_paths(branches):
        """
        Try to find vertex-disjoint paths between all pairs of branches.
        Returns (success, paths) where paths is a dict mapping (u,v) to path.
        Uses greedy assignment with backtracking for small instances.
        """
        if len(branches) < 2:
            return False, {}
        
        branch_set = set(branches)
        k = len(branches)
        pairs = list(combinations(branches, 2))
        num_pairs = len(pairs)
        
        # For each pair, find candidate paths (shortest first)
        # Then try to assign paths such that interiors are disjoint
        
        # Precompute shortest paths for each pair
        pair_paths = {}
        for u, v in pairs:
            if v in adj[u]:
                pair_paths[(u, v)] = [[u, v]]
            else:
                # BFS to find shortest path
                visited = {u}
                queue = deque([(u, [u])])
                paths_found = []
                while queue and len(paths_found) < 5:  # Limit to 5 candidate paths
                    curr, path = queue.popleft()
                    if curr == v:
                        paths_found.append(path)
                        continue
                    for neighbor in sorted(adj[curr]):  # Sorted for determinism
                        if neighbor in visited:
                            continue
                        # Can pass through non-branch vertices, or v (endpoint)
                        if neighbor not in branch_set or neighbor == v:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                pair_paths[(u, v)] = paths_found
        
        # Check if all pairs have at least one path
        for u, v in pairs:
            if not pair_paths[(u, v)]:
                return False, {}
        
        # Greedy assignment: try to assign paths with disjoint interiors
        # Use backtracking for small k
        def backtrack(idx, used_interiors, assignment):
            if time.time() > deadline:
                return None
            if idx == num_pairs:
                return assignment
            
            u, v = pairs[idx]
            for path in pair_paths[(u, v)]:
                interior = set(path[1:-1])
                if not interior.issubset(used_interiors):  # No overlap with used
                    # Check interior vertices are not branch vertices
                    if interior.isdisjoint(branch_set):
                        new_used = used_interiors | interior
                        new_assignment = assignment + [(u, v, path)]
                        result = backtrack(idx + 1, new_used, new_assignment)
                        if result is not None:
                            return result
            return None
        
        result = backtrack(0, set(), [])
        if result is None:
            return False, {}
        
        paths_dict = {}
        for u, v, path in result:
            paths_dict[(u, v)] = path
        
        return True, paths_dict

    # Step 1: Find maximum clique
    max_clique = max_clique_bronk()
    
    if len(max_clique) < 2:
        if edges:
            u, v = edges[0]
            max_clique = [u, v]
        else:
            max_clique = [0, 1]
    
    branches = list(max_clique)
    
    # Step 2: Try to add more vertices (not in clique) if disjoint paths exist
    remaining = [v for v in range(n) if v not in set(branches)]
    
    # Sort remaining by degree (high degree first)
    remaining.sort(key=lambda v: len(adj[v]), reverse=True)
    
    for candidate in remaining:
        if time.time() > deadline - 10:
            break
        new_branches = branches + [candidate]
        success, _ = find_disjoint_paths(new_branches)
        if success:
            branches = new_branches
    
    # Step 3: Get final paths
    success, paths_dict = find_disjoint_paths(branches)
    if not success:
        # Fallback to just the clique
        branches = list(max_clique)
        success, paths_dict = find_disjoint_paths(branches)
        if not success:
            # Last resort: use direct edges for clique
            paths_dict = {}
            for u, v in combinations(branches, 2):
                paths_dict[(u, v)] = [u, v]
    
    # Build output
    paths_list = []
    for u, v in combinations(branches, 2):
        if (u, v) in paths_dict:
            paths_list.append([int(x) for x in paths_dict[(u, v)]])
        elif (v, u) in paths_dict:
            paths_list.append([int(x) for x in paths_dict[(v, u)]])
        else:
            # Should not happen
            paths_list.append([int(u), int(v)])
    
    return {'branches': [int(b) for b in branches], 'paths': paths_list}

# EVOLVE-BLOCK-END
