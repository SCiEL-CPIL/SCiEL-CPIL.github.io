"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [list() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute all simple paths up to length 7 between all pairs
    # path_cache[(u,v)] = list of paths (each path is a list of internal vertices)
    path_cache = {}
    
    def bfs_paths(src, dst, max_len, excluded):
        """Find all simple paths from src to dst with at most max_len edges,
        avoiding vertices in excluded set (except src and dst)."""
        results = []
        # BFS with path tracking
        queue = deque()
        queue.append((src, [src], {src}))
        
        while queue:
            node, path, visited = queue.popleft()
            if len(path) - 1 > max_len:
                continue
            if node == dst:
                if len(path) > 2:  # at least one internal vertex
                    results.append(path[1:-1])
                continue
            for nb in adj_list[node]:
                if nb in visited:
                    continue
                if nb in excluded and nb != src and nb != dst:
                    continue
                new_visited = visited | {nb}
                queue.append((nb, path + [nb], new_visited))
        
        return results
    
    # Precompute paths for all pairs up to length 7
    # This is expensive, so we'll do it lazily
    def get_paths(u, v, excluded, max_len=7):
        key = (u, v) if u < v else (v, u)
        excl_key = frozenset(excluded)
        full_key = (key, excl_key, max_len)
        if full_key not in path_cache:
            if u == v or v in adj[u]:
                path_cache[full_key] = []
            else:
                # Try increasing lengths
                all_paths = []
                for ml in range(2, max_len + 1, 2):  # even lengths: 2,4,6 (1,3,5 internal)
                    paths = bfs_paths(u, v, ml, excluded)
                    all_paths.extend(paths)
                path_cache[full_key] = all_paths
        return path_cache[full_key]
    
    # Precompute length-3 paths (single internal vertex) - fast
    path_3 = [[[] for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if v in adj[u]:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    path_3[u][v].append(w)
    
    def find_subdivision_k3(k, candidates):
        """Find subdivision of size k using only length-1 and length-3 paths."""
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                return None
                
            B = list(combo)
            B_set = set(B)
            
            non_adj_pairs = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v not in adj[u]:
                        non_adj_pairs.append((u, v))
                        
            # Pigeonhole check
            available_vertices = n - k
            if len(non_adj_pairs) > available_vertices:
                continue
                
            # Build matching problem
            pair_to_verts = {}
            for idx, (u, v) in enumerate(non_adj_pairs):
                valid_ws = [w for w in path_3[u][v] if w not in B_set]
                pair_to_verts[idx] = valid_ws
                
            # Check if any pair has no options
            possible = True
            for idx in range(len(non_adj_pairs)):
                if not pair_to_verts[idx]:
                    possible = False
                    break
                    
            if not possible:
                continue
                
            # Bipartite Matching (Kuhn's algorithm)
            match_vertex = {v: -1 for v in range(n)}
            
            def try_kuhn(v, seen):
                for u in pair_to_verts[v]:
                    if seen[u]:
                        continue
                    seen[u] = True
                    if match_vertex[u] == -1 or try_kuhn(match_vertex[u], seen):
                        match_vertex[u] = v
                        return True
                return False
            
            found_matching = True
            for i in range(len(non_adj_pairs)):
                seen = {v: False for v in range(n)}
                if not try_kuhn(i, seen):
                    found_matching = False
                    break
                    
            if not found_matching:
                continue
                
            # Construct result
            pair_idx_map = {}
            for idx, (u, v) in enumerate(non_adj_pairs):
                pair_idx_map[(u, v)] = idx
                pair_idx_map[(v, u)] = idx
            
            paths = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v in adj[u]:
                        paths.append([u, v])
                    else:
                        idx = pair_idx_map[(u, v)]
                        w = None
                        for vert in range(n):
                            if match_vertex[vert] == idx:
                                w = vert
                                break
                        if w is None:
                            return None
                        paths.append([u, w, v])
                        
            return {'branches': B, 'paths': paths}
            
        return None
    
    def find_subdivision_general(k, candidates, max_path_len=7):
        """Find subdivision of size k using paths up to max_path_len."""
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                return None
                
            B = list(combo)
            B_set = set(B)
            
            non_adj_pairs = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v not in adj[u]:
                        non_adj_pairs.append((u, v))
                        
            # Pigeonhole: each non-adj pair needs at least 1 internal vertex
            available_vertices = n - k
            if len(non_adj_pairs) > available_vertices:
                continue
                
            # Get paths for each non-adj pair
            pair_paths = {}
            possible = True
            for idx, (u, v) in enumerate(non_adj_pairs):
                paths = get_paths(u, v, B_set, max_path_len)
                # Filter out paths that use branch vertices as internals
                valid_paths = []
                for p in paths:
                    if not any(w in B_set for w in p):
                        valid_paths.append(p)
                if not valid_paths:
                    possible = False
                    break
                pair_paths[idx] = valid_paths
            
            if not possible:
                continue
            
            # Now we need to assign disjoint internal vertices to each pair
            # This is a more complex constraint satisfaction problem
            # Use backtracking with forward checking
            
            used_vertices = set()
            
            def backtrack(idx):
                if time.time() > deadline:
                    return None
                if idx == len(non_adj_pairs):
                    return {}
                
                u, v = non_adj_pairs[idx]
                for path in pair_paths[idx]:
                    # Check if all internal vertices are available
                    if any(w in used_vertices for w in path):
                        continue
                    # Check if all internal vertices are not branch vertices
                    if any(w in B_set for w in path):
                        continue
                    
                    # Assign this path
                    new_used = used_vertices | set(path)
                    result = backtrack(idx + 1)
                    if result is not None:
                        result[idx] = path
                        return result
                    
                return None
            
            assignment = backtrack(0)
            if assignment is None:
                continue
            
            # Construct result
            pair_idx_map = {}
            for idx, (u, v) in enumerate(non_adj_pairs):
                pair_idx_map[(u, v)] = idx
                pair_idx_map[(v, u)] = idx
            
            paths = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v in adj[u]:
                        paths.append([u, v])
                    else:
                        idx = pair_idx_map[(u, v)]
                        internal = assignment[idx]
                        paths.append([u] + internal + [v])
                        
            return {'branches': B, 'paths': paths}
            
        return None
    
    best_result = None
    best_k = 2
    
    # Strategy 1: Try K3-style (length 1 and 3 paths) for large k
    for k in range(12, 3, -1):
        if time.time() > deadline:
            break
        pool_size = min(16, n)
        candidates = sorted_vertices[:pool_size]
        result = find_subdivision_k3(k, candidates)
        if result:
            best_result = result
            best_k = k
            break
    
    # Strategy 2: Try general paths for medium k
    if best_result is None or best_k < 8:
        for k in range(max(8, best_k), 3, -1):
            if time.time() > deadline:
                break
            pool_size = min(14, n)
            candidates = sorted_vertices[:pool_size]
            result = find_subdivision_general(k, candidates, max_path_len=5)
            if result and (best_result is None or k > best_k):
                best_result = result
                best_k = k
                break
    
    # Strategy 3: Find max clique as fallback
    if best_result is None or best_k < 4:
        def max_clique(vertices):
            if not vertices:
                return []
            best = []
            def backtrack(verts, current):
                nonlocal best
                if len(current) > len(best):
                    best = list(current)
                for i, v in enumerate(verts):
                    current.append(v)
                    backtrack([x for x in verts[i+1:] if x in adj[v]], current)
                    current.pop()
            backtrack(vertices, [])
            return best
            
        clique = max_clique(sorted_vertices[:16])
        if len(clique) >= 2:
            B = clique
            paths = []
            for i in range(len(B)):
                for j in range(i+1, len(B)):
                    paths.append([B[i], B[j]])
            if best_result is None or len(B) > best_k:
                best_result = {'branches': B, 'paths': paths}
                best_k = len(B)
    
    # Fallback to edge
    if best_result is None:
        u, v = P['edges'][0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    # Ensure types are correct
    best_result['branches'] = [int(x) for x in best_result['branches']]
    for i in range(len(best_result['paths'])):
        best_result['paths'][i] = [int(x) for x in best_result['paths'][i]]
        
    return best_result
# EVOLVE-BLOCK-END
