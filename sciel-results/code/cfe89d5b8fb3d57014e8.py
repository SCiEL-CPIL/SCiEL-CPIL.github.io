import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    top_candidates = sorted_verts[:15]

    def find_paths(s, t, blocked, limit=5):
        paths = []
        # DFS to find up to 'limit' simple paths
        # Depth limit to prevent explosion
        def dfs(u, path, visited, depth):
            if len(paths) >= limit:
                return
            if depth > 10: # Heuristic depth limit
                return
            if u == t:
                paths.append(path[:])
                return
            for v in adj[u]:
                if v not in visited and v not in blocked:
                    visited.add(v)
                    path.append(v)
                    dfs(v, path, visited, depth + 1)
                    path.pop()
                    visited.remove(v)
        
        dfs(s, [s], {s}, 0)
        paths.sort(key=len)
        return paths

    def check_subdivision(branches):
        k = len(branches)
        pairs = list(combinations(range(k), 2))
        
        def backtrack(idx, used_interiors, current_paths):
            if time.time() > deadline:
                return None
            if idx == len(pairs):
                return current_paths
            
            i, j = pairs[idx]
            u, v = branches[i], branches[j]
            
            # Blocked vertices: previously used interiors + all other branches
            other_branches = set(branches) - {u, v}
            blocked = used_interiors | other_branches
            
            candidates = find_paths(u, v, blocked, limit=5)
            
            for path in candidates:
                if len(path) < 2: continue
                interiors = set(path[1:-1])
                # interiors should not intersect used_interiors (guaranteed by blocked)
                # and should not contain other branches (guaranteed by blocked)
                new_used = used_interiors | interiors
                new_paths = current_paths + [path]
                res = backtrack(idx + 1, new_used, new_paths)
                if res is not None:
                    return res
            return None
        
        return backtrack(0, set(), [])

    best_branches = None
    best_paths = None
    best_k = 0

    # Try k from 6 down to 4
    for k in range(6, 3, -1):
        if time.time() > deadline - 5:
            break
        found = False
        for combo in combinations(top_candidates, k):
            if time.time() > deadline - 5:
                break
            branches = list(combo)
            paths = check_subdivision(branches)
            if paths is not None:
                best_branches = branches
                best_paths = paths
                best_k = k
                found = True
                break
        if found:
            break

    # If no k>=4, try k=3 (triangle)
    if best_k < 3:
        es = set()
        for u, v in edges:
            es.add((u, v))
            es.add((v, u))
        
        for i in range(n):
            if time.time() > deadline - 1:
                break
            for j in range(i + 1, n):
                if (i, j) not in es:
                    continue
                for k in range(j + 1, n):
                    if (j, k) in es and (i, k) in es:
                        best_branches = [i, j, k]
                        best_paths = [[i, j], [j, k], [i, k]]
                        best_k = 3
                        break
                if best_k == 3:
                    break
            if best_k == 3:
                break

    # Fallback k=2
    if best_k < 2:
        if edges:
            u, v = edges[0]
            best_branches = [u, v]
            best_paths = [[u, v]]
        else:
            best_branches = [0, 1]
            best_paths = [[0, 1]]

    return {
        'branches': list(best_branches),
        'paths': [list(p) for p in best_paths]
    }
