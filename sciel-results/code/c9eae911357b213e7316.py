import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
        
    def is_edge(u, v):
        return (min(u,v), max(u,v)) in edge_set

    candidates = {}
    for u in range(n):
        candidates[u] = {}
        for v in range(u+1, n):
            paths = []
            if is_edge(u, v):
                paths.append([u, v])
            for w in range(n):
                if w != u and w != v and is_edge(u, w) and is_edge(w, v):
                    paths.append([u, w, v])
            for w in range(n):
                if w == u or w == v: continue
                if not is_edge(u, w): continue
                for z in range(n):
                    if z == u or z == v or z == w: continue
                    if is_edge(w, z) and is_edge(z, v):
                        paths.append([u, w, z, v])
            candidates[u][v] = paths
            candidates[v][u] = paths

    def check_subset(B):
        B_set = set(B)
        pairs = list(itertools.combinations(B, 2))
        pairs.sort(key=lambda p: len(candidates[p[0]][p[1]]))
        
        paths = {}
        used_interiors = set()
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            for path in candidates[u][v]:
                valid = True
                new_interiors = set()
                for node in path[1:-1]:
                    if node in B_set or node in used_interiors:
                        valid = False
                        break
                    new_interiors.add(node)
                
                if not valid:
                    continue
                
                paths[(u, v)] = path
                used_interiors.update(new_interiors)
                
                if backtrack(idx + 1):
                    return True
                
                paths.pop((u, v))
                used_interiors.difference_update(new_interiors)
            
            return False
        
        if backtrack(0):
            res_paths = [paths[(u, v)] for u, v in pairs]
            return True, res_paths
        return False, None

    def find_max_clique():
        best_clique = []
        
        def extend_clique(candidates, current):
            nonlocal best_clique
            if time.time() > deadline:
                return
            if len(current) > len(best_clique):
                best_clique = list(current)
            if not candidates:
                return
            
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            
            new_candidates = [u for u in candidates if u != v and is_edge(u, v)]
            extend_clique(new_candidates, current + [v])
            
            extend_clique([u for u in candidates if u != v], current)
            
        extend_clique(list(range(n)), [])
        return best_clique

    max_clique = find_max_clique()
    best_k = len(max_clique)
    best_branches = max_clique
    best_paths = []
    if best_k >= 2:
        _, best_paths = check_subset(best_branches)
        
    for k in range(n, best_k, -1):
        if time.time() > deadline - 10: break
        
        if k <= 10:
            for B in itertools.combinations(range(n), k):
                if time.time() > deadline - 5: break
                ok, paths = check_subset(list(B))
                if ok:
                    return {'branches': list(B), 'paths': paths}
            
    if best_k >= 2:
        return {'branches': best_branches, 'paths': best_paths}
    
    return {'branches': [0, 1], 'paths': [[0, 1]]}
