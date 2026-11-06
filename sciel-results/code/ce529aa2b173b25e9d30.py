import numpy as np

def construct():
    q = 7
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n_total = len(points)
    
    adj = [set() for _ in range(n_total)]
    for i in range(n_total):
        u = points[i]
        for j in range(i + 1, n_total):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    def try_find_subset(n_target):
        k = n_total - n_target
        min_deg_req = n_target - 39
        # D = 8 - min_deg_req
        # For n=46, min_deg=7, D=1.
        # Condition: |N(u) \cap R| <= 1 for all u not in R.
        # This is equivalent to: For any v1, v2 in R, N(v1) \cap N(v2) \setminus {v1, v2} \subseteq R.
        
        if k > n_total:
            return None
            
        import random
        
        # Try a few seeds to increase robustness
        for seed in [42, 123, 999, 777]:
            random.seed(seed)
            
            def backtrack(R, in_R, count):
                if count == k:
                    return R
                
                candidates = []
                for v in range(n_total):
                    if v in in_R:
                        continue
                    
                    valid = True
                    for w in R:
                        common = adj[v] & adj[w]
                        for u in common:
                            if u not in in_R:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        candidates.append(v)
                
                if not candidates:
                    return None
                    
                random.shuffle(candidates)
                
                for v in candidates:
                    new_R = R + [v]
                    new_in_R = in_R | {v}
                    
                    res = backtrack(new_R, new_in_R, count + 1)
                    if res is not None:
                        return res
                return None
                
            initial_R = []
            initial_in_R = set()
            res = backtrack(initial_R, initial_in_R, 0)
            
            if res is not None:
                R = set(res)
                V_remaining = [i for i in range(n_total) if i not in R]
                
                # Verify
                valid = True
                for u in V_remaining:
                    deg = len([v for v in adj[u] if v not in R])
                    if deg < min_deg_req:
                        valid = False
                        break
                if valid:
                    return V_remaining, R

        return None

    for n_target in [46, 45, 44, 43]:
        res = try_find_subset(n_target)
        if res is not None:
            V_remaining, R = res
            n = len(V_remaining)
            
            old_to_new = {old: new for new, old in enumerate(V_remaining)}
            
            edges = []
            for u in V_remaining:
                for v in adj[u]:
                    if v in old_to_new:
                        if old_to_new[u] < old_to_new[v]:
                            edges.append([old_to_new[u], old_to_new[v]])
            
            return {'n': n, 'edges': edges}
            
    return {'n': 39, 'edges': []}
