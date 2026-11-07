import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145.0
    trees = instance_trees(20, 611)
    N = 20
    all_edges = set()
    for a in range(N):
        for b in range(a + 1, N):
            all_edges.add((a, b))
    
    def edge_key(u, v):
        return (u, v) if u < v else (v, u)
    
    # Precompute tree structures
    tree_info = []
    for i in range(20):
        k = i + 1
        edges = trees[i]
        if k == 1:
            tree_info.append({'k': 1, 'edges': [], 'adj': [[]], 'root_order': [0], 'parent': [-2]})
            continue
        if k == 2:
            tree_info.append({'k': 2, 'edges': [(0, 1)], 'adj': [[1], [0]], 'root_order': [0, 1], 'parent': [-2, 0]})
            continue
        adj = [[] for _ in range(k)]
        for (u, v) in edges:
            adj[u].append(v); adj[v].append(u)
        # BFS from 0
        parent = [-1] * k
        parent[0] = -2
        order = [0]
        q = deque([0])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if parent[y] == -1:
                    parent[y] = x
                    q.append(y)
                    order.append(y)
        tree_info.append({'k': k, 'edges': edges, 'adj': adj, 'root_order': order, 'parent': parent})
    
    def try_pack(seed_val):
        rng = Random(seed_val)
        used = set()
        result = [None] * 20
        
        # Process trees in order of decreasing size for constraint propagation
        order_idx = list(range(20))
        order_idx.sort(key=lambda i: -(i + 1))
        
        for idx in order_idx:
            info = tree_info[idx]
            k = info['k']
            
            if k == 1:
                result[idx] = [0]
                continue
            
            if k == 2:
                # Find any unused edge
                placed = False
                for a in range(N):
                    for b in range(a + 1, N):
                        e = (a, b)
                        if e not in used:
                            result[idx] = [a, b]
                            used.add(e)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    return None
                continue
            
            # General tree: backtracking with MRV heuristic
            adj = info['adj']
            parent = info['parent']
            order = info['root_order']
            
            assignment = [-1] * k
            used_hosts = [False] * N
            
            def dfs(idx_v):
                if time.time() > deadline:
                    return False
                if idx_v == k:
                    return True
                
                v = order[idx_v]
                pv = parent[v]
                
                # Get candidate hosts
                candidates = []
                for h in range(N):
                    if used_hosts[h]:
                        continue
                    if pv >= 0:
                        ph = assignment[pv]
                        e = edge_key(h, ph)
                        if e in used:
                            continue
                    candidates.append(h)
                
                if not candidates:
                    return False
                
                # Heuristic: prefer vertices with fewer available edges (more constrained)
                # Count how many free edges each candidate has
                def count_free(h):
                    cnt = 0
                    for nb in range(N):
                        if nb == h:
                            continue
                        e = edge_key(h, nb)
                        if e not in used and not used_hosts[nb]:
                            cnt += 1
                    return cnt
                
                # Shuffle for randomness but bias toward more constrained
                rng.shuffle(candidates)
                
                for h in candidates:
                    assignment[v] = h
                    used_hosts[h] = True
                    if dfs(idx_v + 1):
                        return True
                    used_hosts[h] = False
                
                assignment[v] = -1
                return False
            
            if dfs(0):
                result[idx] = [assignment[i] for i in range(k)]
                for (u, v) in info['edges']:
                    used.add(edge_key(assignment[u], assignment[v]))
            else:
                return None
        
        # Verify
        seen = set()
        for i in range(20):
            r = result[i]
            if r is None:
                if i == 0:
                    continue
                return None
            if len(set(r)) != len(r):
                return None
            for x in r:
                if not (0 <= x <= 19):
                    return None
            for (u, v) in trees[i]:
                e = edge_key(r[u], r[v])
                if e in seen:
                    return None
                seen.add(e)
        
        return result
    
    # Try multiple seeds
    for s in range(500):
        if time.time() > deadline:
            break
        res = try_pack(s)
        if res is not None:
            out = []
            for i in range(20):
                r = res[i]
                if r is None:
                    out.append(None)
                else:
                    out.append([int(x) for x in r])
            return out
    
    # Fallback: try a deterministic greedy approach
    used = set()
    result = [None] * 20
    order_idx = list(range(20))
    order_idx.sort(key=lambda i: -(i + 1))
    
    for idx in order_idx:
        info = tree_info[idx]
        k = info['k']
        if k == 1:
            result[idx] = [0]
            continue
        if k == 2:
            for a in range(N):
                for b in range(a + 1, N):
                    e = (a, b)
                    if e not in used:
                        result[idx] = [a, b]
                        used.add(e)
                        break
                if result[idx]:
                    break
            if not result[idx]:
                return [None] * 19 + [list(range(20))]
            continue
        # Greedy: assign in BFS order, pick first available
        adj = info['adj']
        parent = info['parent']
        order = info['root_order']
        assignment = [-1] * k
        used_hosts = [False] * N
        success = True
        for idx_v in range(k):
            v = order[idx_v]
            pv = parent[v]
            placed = False
            for h in range(N):
                if used_hosts[h]:
                    continue
                if pv >= 0:
                    ph = assignment[pv]
                    e = edge_key(h, ph)
                    if e in used:
                        continue
                assignment[v] = h
                used_hosts[h] = True
                placed = True
                break
            if not placed:
                success = False
                break
        if success:
            result[idx] = [assignment[i] for i in range(k)]
            for (u, v) in info['edges']:
                used.add(edge_key(assignment[u], assignment[v]))
        else:
            return [None] * 19 + [list(range(20))]
    
    # Verify greedy result
    seen = set()
    for i in range(20):
        r = result[i]
        if r is None:
            return [None] * 19 + [list(range(20))]
        if len(set(r)) != len(r):
            return [None] * 19 + [list(range(20))]
        for (u, v) in trees[i]:
            e = edge_key(r[u], r[v])
            if e in seen:
                return [None] * 19 + [list(range(20))]
            seen.add(e)
    
    return [[int(x) for x in r] if r is not None else None for r in result]
