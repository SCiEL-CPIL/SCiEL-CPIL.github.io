import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set
    adj = {i: [] for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    # Calculate degrees
    degrees = {i: len(adj[i]) for i in range(n)}
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if (vertices[i], vertices[j]) not in edge_set:
                    return False
        return True
        
    def find_subdivision(branches):
        # branches is a list of vertices
        k = len(branches)
        if k < 2:
            return None
            
        branch_set = set(branches)
        branch_mask = 0
        for b in branches:
            branch_mask |= (1 << b)
            
        pairs = list(combinations(branches, 2))
        
        # Order pairs: maybe those with fewer common neighbors first?
        # For now, fixed order.
        
        # Pre-compute something? No, dynamic used set.
        
        # Backtracking search
        # We use a recursive function
        # To avoid global state issues, we use closures or pass state
        
        # We want to find paths for all pairs such that interiors are disjoint.
        
        # Heuristic: Sort pairs by the number of available neighbors?
        # Let's just use the given order.
        
        def search(pair_idx, used_mask):
            if pair_idx == len(pairs):
                return True
                
            u, v = pairs[pair_idx]
            
            # Find candidate paths from u to v
            # Avoid used_mask and branch_set (except u, v)
            # We will use DFS to find up to 10 paths
            # To prioritize short paths, we can use BFS depth limit or just DFS
            # Let's use a simple DFS that yields paths
            
            paths_found = []
            # To ensure we find short paths, we can sort neighbors by distance?
            # Or just try all.
            
            def gen_paths(cur, target, path, visited):
                if len(paths_found) >= 10:
                    return
                if cur == target:
                    paths_found.append(list(path))
                    return
                for nei in adj[cur]:
                    # Check if nei is allowed
                    # If nei is in used_mask, skip
                    if (visited & (1 << nei)):
                        continue
                    # If nei is a branch and not target, skip
                    if (branch_mask & (1 << nei)) and nei != target:
                        continue
                    
                    visited |= (1 << nei)
                    path.append(nei)
                    gen_paths(nei, target, path, visited)
                    path.pop()
                    visited &= ~(1 << nei)
                    if len(paths_found) >= 10:
                        return
            
            # Initialize visited
            # u is start, so it's "visited" in the path sense, but we start there.
            # We need to prevent going back to u? No, simple path.
            # visited should contain u?
            # If we start at u, we mark u as visited to prevent cycles.
            init_visited = used_mask | (1 << u)
            # Note: branch_mask is handled in the loop check, but u is a branch.
            # The check `if (branch_mask & (1 << nei)) and nei != target` handles other branches.
            # u is in branch_mask, but we start there.
            # If we go to a neighbor, and that neighbor is u? No, simple graph.
            
            gen_paths(u, v, [u], init_visited)
            
            for p in paths_found:
                interior = p[1:-1]
                new_used = used_mask
                for x in interior:
                    new_used |= (1 << x)
                
                if search(pair_idx + 1, new_used):
                    return True
            return False
            
        if search(0, 0):
            # We found a valid subdivision, but we didn't store the paths!
            # We need to store them.
            # Let's modify search to return the paths or store them globally.
            pass
        return None # Placeholder

    # The above search doesn't return the paths. We need to fix that.
    # Let's rewrite the search to collect paths.
    
    def find_subdivision_with_paths(branches):
        k = len(branches)
        if k < 2:
            return None
            
        branch_set = set(branches)
        branch_mask = 0
        for b in branches:
            branch_mask |= (1 << b)
            
        pairs = list(combinations(branches, 2))
        paths_dict = {}
        
        def search(pair_idx, used_mask):
            if pair_idx == len(pairs):
                return True
                
            u, v = pairs[pair_idx]
            
            paths_found = []
            
            def gen_paths(cur, target, path, visited):
                if len(paths_found) >= 10:
                    return
                if cur == target:
                    paths_found.append(list(path))
                    return
                for nei in adj[cur]:
                    if (visited & (1 << nei)):
                        continue
                    if (branch_mask & (1 << nei)) and nei != target:
                        continue
                    
                    visited |= (1 << nei)
                    path.append(nei)
                    gen_paths(nei, target, path, visited)
                    path.pop()
                    visited &= ~(1 << nei)
                    if len(paths_found) >= 10:
                        return
            
            init_visited = used_mask | (1 << u)
            gen_paths(u, v, [u], init_visited)
            
            for p in paths_found:
                interior = p[1:-1]
                new_used = used_mask
                for x in interior:
                    new_used |= (1 << x)
                
                if search(pair_idx + 1, new_used):
                    paths_dict[(u, v)] = p
                    return True
            return False
            
        if search(0, 0):
            # Construct the final paths list
            final_paths = []
            for u, v in combinations(branches, 2):
                # paths_dict might have (v, u) or (u, v)
                if (u, v) in paths_dict:
                    final_paths.append(paths_dict[(u, v)])
                else:
                    final_paths.append(paths_dict[(v, u)])
            return final_paths
        return None

    # Main loop
    best_result = None
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # Try k from high to low
    # Max possible k is n, but practically small.
    # We'll try up to 8.
    max_k_to_try = min(n, 8)
    
    for k in range(max_k_to_try, 1, -1):
        if time.time() > deadline:
            break
            
        # Determine candidate pool
        if k <= 3:
            pool_size = n
        elif k == 4:
            pool_size = 20
        elif k == 5:
            pool_size = 15
        elif k == 6:
            pool_size = 12
        else:
            pool_size = 10
            
        candidates = sorted_vertices[:pool_size]
        
        for comb in combinations(candidates, k):
            if time.time() > deadline:
                break
                
            # Check if clique
            if is_clique(comb):
                # Paths are just edges
                paths = []
                for u, v in combinations(comb, 2):
                    paths.append([u, v])
                return {'branches': list(comb), 'paths': paths}
            
            # Check for subdivision
            res = find_subdivision_with_paths(list(comb))
            if res is not None:
                return {'branches': list(comb), 'paths': res}
                
    # Fallback: return a valid 2-branch solution (an edge)
    u, v = edges[0]
    return {'branches': [u, v], 'paths': [[u, v]]}

# EVOLVE-BLOCK-END
