import time
from itertools import combinations

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

    def find_simple_paths(u, v, banned, limit=10):
        """Find up to 'limit' simple paths from u to v avoiding 'banned' vertices in interior."""
        paths = []
        if u in banned or v in banned:
            return paths
        if u == v:
            return [[u]]
            
        # DFS to find paths
        # To avoid exponential blowup, we can use BFS for shortest, then DFS for others?
        # Given small graph, DFS with pruning is okay.
        # We want short paths first.
        
        def dfs(cur, path, visited):
            if time_up():
                return
            if len(paths) >= limit:
                return
            if cur == v:
                paths.append(list(path))
                return
            
            for nb in adj[cur]:
                if nb == u and len(path) > 1: # Don't go back to start immediately? No, simple path means no repeated vertices.
                    continue
                if nb in visited:
                    continue
                if nb in banned and nb != v: # Endpoints allowed, but banned set usually contains branches. 
                    # In our use case, 'banned' contains all branches. 
                    # u and v are branches. So u, v are in banned.
                    # But we allow u and v as endpoints.
                    # Interior vertices must not be in banned.
                    # So if nb is in banned and nb != v (and nb != u, but u is start), skip.
                    # Wait, if nb is a branch other than v, we can't use it.
                    # So: if nb in banned and nb != v: skip.
                    # Also, if nb == u, we skip because of visited.
                    pass
                
                # Correct check:
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

    def try_add_vertex(v, branches):
        """Try to add vertex v to existing branches (which form a clique subdivision, e.g., a clique)."""
        if time_up():
            return None
            
        targets = list(branches)
        # We need paths from v to each target.
        # The interiors must be disjoint and avoid all branches (including v).
        banned = set(branches) | {v}
        
        # Order targets to make search easier? 
        # Maybe targets with fewer available neighbors first?
        # For now, just try all permutations of targets? 4! = 24.
        # Or just try in given order and rely on backtracking on path choice.
        
        def search(k, used_interiors):
            if time_up():
                return None
            if k == len(targets):
                return True # All paths found
            
            target = targets[k]
            # Find candidate paths from v to target
            # Avoid 'banned' in interior, and avoid 'used_interiors'
            # Note: 'banned' already contains all branches.
            # 'used_interiors' contains vertices used in previous paths.
            
            # We need to find paths that avoid (banned \ {v, target}) U used_interiors
            # Actually, 'banned' contains v and target.
            # Interior cannot be in banned. So interior cannot be v or target (already true for simple path)
            # and cannot be any other branch.
            # Interior cannot be in used_interiors.
            
            effective_banned = set(banned) | used_interiors
            
            # Find paths
            candidates = find_simple_paths(v, target, effective_banned, limit=5)
            
            for path in candidates:
                new_used = set(used_interiors)
                for node in path[1:-1]:
                    new_used.add(node)
                    
                res = search(k + 1, new_used)
                if res is not None:
                    # Store the path
                    return path # This logic is slightly flawed for returning all paths, let's fix structure
                    # Better: return a list of paths or modify a global list
                    
            return None
            
        # To properly return the paths, we need a different structure.
        # Let's use a global list for the current solution attempt.
        solution_paths = []
        
        def search_paths(k, used_interiors):
            if time_up():
                return False
            if k == len(targets):
                return True
            
            target = targets[k]
            effective_banned = set(banned) | used_interiors
            candidates = find_simple_paths(v, target, effective_banned, limit=5)
            
            for path in candidates:
                new_used = set(used_interiors)
                for node in path[1:-1]:
                    new_used.add(node)
                
                solution_paths.append(path)
                if search_paths(k + 1, new_used):
                    return True
                solution_paths.pop()
                
            return False
            
        if search_paths(0, set()):
            return solution_paths
        return None

    # 1. Find a K4
    best_k4 = None
    for combo in combinations(range(n), 4):
        if time_up():
            break
        is_clique = True
        for u, v in combinations(combo, 2):
            if (u, v) not in edge_set:
                is_clique = False
                break
        if is_clique:
            best_k4 = list(combo)
            break # Just take the first one found
            
    if best_k4 is None:
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
                paths = [list(e) for e in combinations(branches, 2)]
                return {'branches': branches, 'paths': paths}
        
        # Fallback to edge
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    # 2. Try to expand K4 to size 5
    branches = best_k4
    for v in range(n):
        if time_up():
            break
        if v in branches:
            continue
            
        # Quick check: v must be connected to at least one branch? 
        # Actually, paths can be long. But v needs to be in the same component.
        # The graph is likely connected.
        
        paths_to_add = try_add_vertex(v, branches)
        if paths_to_add is not None:
            # We found a valid subdivision of size 5
            new_branches = branches + [v]
            all_paths = []
            # Add original K4 edges
            for u, w in combinations(branches, 2):
                all_paths.append([u, w])
            # Add new paths
            all_paths.extend(paths_to_add)
            
            return {'branches': new_branches, 'paths': all_paths}
            
    # 3. Return the K4
    branches = best_k4
    paths = [list(e) for e in combinations(branches, 2)]
    return {'branches': branches, 'paths': paths}
