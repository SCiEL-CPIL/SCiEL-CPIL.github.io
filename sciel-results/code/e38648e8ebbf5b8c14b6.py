import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    N = 20
    full_mask = (1 << N) - 1
    start_time = time.time()
    TIMEOUT = 115.0
    
    trees = instance_trees(N)
    
    # Preprocess
    tree_info = []
    for k in range(2, N + 1):
        edges = trees[k - 1]
        adj = {i: [] for i in range(k)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        root = 0
        visited = {root}
        order = [root]
        parent = {root: -1}
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        
        tree_info.append({
            'k': k,
            'order': order,
            'parent': parent,
            'root': root
        })
    
    tree_info.sort(key=lambda x: x['k'], reverse=True)
    
    used_edges = [0] * N
    host_map = {k: [-1] * k for k in range(2, N + 1)}
    embedded_flags = {k: False for k in range(2, N + 1)}
    
    def popcount(x):
        return bin(x).count('1')
    
    def avail_deg(host):
        return popcount(full_mask ^ used_edges[host])
    
    def solve_backtrack(idx):
        if time.time() - start_time > TIMEOUT:
            return False
        if idx == len(tree_info):
            return True
        
        info = tree_info[idx]
        k = info['k']
        order = info['order']
        parent = info['parent']
        
        local_host_map = [-1] * k
        local_used_hosts = [False] * N
        
        def assign(pos):
            if time.time() - start_time > TIMEOUT:
                return False
            if pos == k:
                for v in range(k):
                    host_map[k][v] = local_host_map[v]
                embedded_flags[k] = True
                return solve_backtrack(idx + 1)
            
            v = order[pos]
            
            if pos == 0:
                if idx == 0:
                    cands = [0]
                else:
                    cands = [h for h in range(N) if not local_used_hosts[h]]
            else:
                p = parent[v]
                ph = local_host_map[p]
                mask_ph = used_edges[ph]
                cands = []
                for h in range(N):
                    if not local_used_hosts[h]:
                        if not (mask_ph & (1 << h)):
                            cands.append(h)
            
            if not cands:
                return False
            
            cands.sort(key=lambda h: avail_deg(h))
            
            for h in cands:
                local_host_map[v] = h
                local_used_hosts[h] = True
                
                if pos > 0:
                    p = parent[v]
                    ph = local_host_map[p]
                    used_edges[ph] |= (1 << h)
                    used_edges[h] |= (1 << ph)
                    
                    if assign(pos + 1):
                        return True
                    
                    used_edges[ph] ^= (1 << h)
                    used_edges[h] ^= (1 << ph)
                
                local_used_hosts[h] = False
                local_host_map[v] = -1
            
            return False
        
        return assign(0)
    
    success = solve_backtrack(0)
    
    if not success:
        # Fallback: Greedy
        used_edges = [0] * N
        host_map = {k: [-1] * k for k in range(2, N + 1)}
        embedded_flags = {k: False for k in range(2, N + 1)}
        
        for info in tree_info:
            k = info['k']
            order = info['order']
            parent = info['parent']
            local_host_map = [-1] * k
            local_used_hosts = [False] * N
            ok = True
            
            for pos, v in enumerate(order):
                if pos == 0:
                    if k == 20: # First tree
                        cands = [0]
                    else:
                        cands = [h for h in range(N) if not local_used_hosts[h]]
                else:
                    p = parent[v]
                    ph = local_host_map[p]
                    mask_ph = used_edges[ph]
                    cands = []
                    for h in range(N):
                        if not local_used_hosts[h]:
                            if not (mask_ph & (1 << h)):
                                cands.append(h)
                
                if not cands:
                    ok = False
                    break
                
                cands.sort(key=lambda h: avail_deg(h))
                h = cands[0]
                
                local_host_map[v] = h
                local_used_hosts[h] = True
                
                if pos > 0:
                    p = parent[v]
                    ph = local_host_map[p]
                    used_edges[ph] |= (1 << h)
                    used_edges[h] |= (1 << ph)
            
            if ok:
                for v in range(k):
                    host_map[k][v] = local_host_map[v]
                embedded_flags[k] = True
    
    # Build result
    result = [None] * N
    result[0] = [0]
    
    for k in range(2, N + 1):
        if embedded_flags.get(k, False):
            result[k - 1] = [int(host_map[k][v]) for v in range(k)]
        else:
            result[k - 1] = None
            
    return result

