import random
import time

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    degrees = [len(adj[i]) for i in range(n)]
    
    def find_path_safe(src, dst, forbidden_internal):
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        queue = [src]
        while queue:
            u = queue.pop(0)
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for v in adj[u]:
                if v not in visited:
                    if v == dst:
                        visited.add(v)
                        parent[v] = u
                        queue.append(v)
                    elif v not in forbidden_internal:
                        visited.add(v)
                        parent[v] = u
                        queue.append(v)
        return None

    def try_greedy(order):
        branches = []
        used_int = set()
        paths = {}
        
        def try_add_branch(v):
            nonlocal branches, used_int, paths
            if not branches:
                branches.append(v)
                return True
            
            local_forbidden = used_int | set(branches)
            
            def backtrack_safe(idx, current_paths, local_forb):
                if idx == len(branches):
                    return True
                
                u = branches[idx]
                path = find_path_safe(v, u, local_forb)
                if path is None:
                    return False
                
                interior = set(path[1:-1])
                if interior & local_forb:
                    return False
                
                new_local_forb = local_forb | interior
                current_paths[u] = path
                
                if backtrack_safe(idx + 1, current_paths, new_local_forb):
                    return True
                
                del current_paths[u]
                return False

            current_paths = {}
            if backtrack_safe(0, current_paths, local_forbidden):
                for u, path in current_paths.items():
                    paths[(v, u)] = path
                    paths[(u, v)] = path
                    used_int.update(set(path[1:-1]))
                branches.append(v)
                return True
            return False

        for v in order:
            try_add_branch(v)
            
        if len(branches) < 2:
            return None, None
            
        out_paths = []
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                u = branches[i]
                v = branches[j]
                if (u, v) in paths:
                    out_paths.append(list(paths[(u, v)]))
                elif (v, u) in paths:
                    out_paths.append(list(paths[(v, u)]))
                else:
                    out_paths.append([u, v])
                    
        return branches, out_paths

    best_branches = []
    best_paths = []
    
    order1 = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    b, p = try_greedy(order1)
    if b and len(b) > len(best_branches):
        best_branches = b
        best_paths = p
        
    start_time = time.time()
    for _ in range(100):
        if time.time() - start_time > 50:
            break
        order2 = list(range(n))
        random.shuffle(order2)
        b, p = try_greedy(order2)
        if b and len(b) > len(best_branches):
            best_branches = b
            best_paths = p

    if not best_branches:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    return {'branches': list(best_branches), 'paths': best_paths}
