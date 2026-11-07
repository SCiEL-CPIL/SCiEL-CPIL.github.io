import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Function to find shortest path
    def shortest_path(s, t, blocked):
        from collections import deque
        if s == t: return []
        if s in blocked or t in blocked: return None
        queue = deque([(s, [s])])
        visited = {s}
        while queue:
            curr, path = queue.popleft()
            if curr == t:
                return path
            for nxt in adj[curr]:
                if nxt not in visited and nxt not in blocked:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    # Function to find k-shortest simple paths (approx)
    def find_paths(s, t, blocked, limit=5):
        paths = []
        from collections import deque
        
        # 1. Find shortest path using BFS
        queue = deque([(s, [s])])
        visited = {s}
        while queue:
            curr, path = queue.popleft()
            if curr == t:
                paths.append(path)
                break
            for nxt in adj[curr]:
                if nxt not in visited and nxt not in blocked:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        
        if len(paths) >= limit:
            return paths
        
        # 2. If more paths are needed, use DFS with a node exploration limit
        # to prevent timeouts.
        explored = 0
        max_explore = 2000 # Limit DFS expansion
        
        def dfs(u, path, visited):
            nonlocal explored
            if len(paths) >= limit or explored > max_explore:
                return
            if u == t:
                # Avoid duplicates
                if path not in paths:
                    paths.append(path[:])
                return
            
            for v in adj[u]:
                if v not in visited and v not in blocked:
                    visited.add(v)
                    path.append(v)
                    explored += 1
                    dfs(v, path, visited)
                    path.pop()
                    visited.remove(v)
                    if len(paths) >= limit or explored > max_explore:
                        return
        
        dfs(s, [s], {s})
        
        # Sort by length to prioritize shorter paths in backtracking
        paths.sort(key=len)
        return paths

    def check_subdivision(branches):
        k = len(branches)
        pairs = list(combinations(range(k), 2))
        # Order pairs: maybe sort by distance in original graph?
        # For now, just use the order generated.
        
        def backtrack(idx, used_interiors, current_paths):
            if time.time() > deadline:
                return None
            if idx == len(pairs):
                return current_paths
            
            i, j = pairs[idx]
            u, v = branches[i], branches[j]
            
            # Find candidate paths
            # We want paths that don't use 'used_interiors'
            # 'used_interiors' contains vertices, not just edges.
            # The path must be simple and not touch any vertex in 'used_interiors'.
            # Note: u and v are branches, so they are allowed to be endpoints, 
            # but they are not in 'used_interiors' (which only contains interiors).
            
            candidates = find_paths(u, v, used_interiors, limit=5)
            
            for path in candidates:
                if len(path) < 2: continue
                interiors = set(path[1:-1])
                if interiors & used_interiors:
                    continue
                new_used = used_interiors | interiors
                new_paths = current_paths + [path]
                res = backtrack(idx + 1, new_used, new_paths)
                if res is not None:
                    return res
            return None
        
        return backtrack(0, set(), [])

    best_branches = None
    best_paths = None
    best_k = 2 # Fallback to edge
    
    # Try to find large cliques subdivisions
    # Candidate vertices: top degrees
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    top_candidates = sorted_verts[:15]
    
    for k in range(6, 2, -1): # Try 6, 5, 4
        if time.time() > deadline - 5: break
        
        # Generate combinations from top candidates
        # To avoid too many combinations, we can sample or limit
        # If k=4, C(15,4) = 1365. Manageable.
        # If k=5, C(15,5) = 3003. Manageable.
        # If k=6, C(15,6) = 5005. Manageable.
        
        found = False
        for combo in combinations(top_candidates, k):
            if time.time() > deadline - 5: break
            branches = list(combo)
            paths = check_subdivision(branches)
            if paths is not None:
                best_branches = branches
                best_paths = paths
                best_k = k
                found = True
                break # Found one for this k, move to next k? 
                # Actually, we want the max k. Since we iterate k down, 
                # the first found is the max.
        
        if found:
            break
            
    # If no k>=4 found, try to find a triangle (k=3)
    if best_k < 3:
        # Simple triangle search
        es = set()
        for u, v in edges:
            es.add((u,v))
            es.add((v,u))
        
        for i in range(n):
            if time.time() > deadline - 1: break
            for j in range(i+1, n):
                if (i,j) not in es: continue
                for k in range(j+1, n):
                    if (j,k) in es and (i,k) in es:
                        best_branches = [i, j, k]
                        best_paths = [[i, j], [j, k], [i, k]]
                        best_k = 3
                        break
                if best_k == 3: break
            if best_k == 3: break

    # Fallback: just an edge
    if best_k < 2 or best_paths is None:
        if edges:
            u, v = edges[0]
            best_branches = [u, v]
            best_paths = [[u, v]]
        else:
            # Graph with no edges? Problem states "pinned graph", likely connected.
            # But to be safe:
            best_branches = [0, 1]
            best_paths = [[0, 1]]

    # Convert to required types
    return {
        'branches': list(best_branches),
        'paths': [list(p) for p in best_paths]
    }
