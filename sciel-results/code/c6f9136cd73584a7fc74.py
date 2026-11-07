"""Rewritten program: attempt exact decomposition of K20 into trees T1..T20 using backtracking."""
from random import Random
import sys
sys.setrecursionlimit(100000)

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
    trees = instance_trees(20, 611)
    n = 20
    
    # Precompute tree adjacency
    tree_adj = []
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in trees[i]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Define several orders to try
    orders = [
        list(range(n-1, -1, -1)), # Largest first
        list(range(n)), # Smallest first
    ]
    
    # Add random orders
    rng_global = Random(999)
    for _ in range(5):
        o = list(range(n))
        rng_global.shuffle(o)
        orders.append(o)
        
    best_value = 0
    best_result = [None] * n
    
    max_trials_per_order = 20
    
    for order in orders:
        for trial in range(max_trials_per_order):
            rng = Random(trial * 1000 + len(order))
            used_edges = set()
            result = [None] * n
            success = True
            
            for idx in order:
                k = idx + 1
                if k == 1:
                    result[idx] = [rng.randint(0, n-1)]
                    continue
                    
                adj = tree_adj[idx]
                mapping = [-1] * k
                host_used = [False] * n
                
                def dfs(depth):
                    if depth == k:
                        return True
                    
                    # Select next node to assign
                    unassigned = [v for v in range(k) if mapping[v] == -1]
                    if not unassigned:
                        return False
                    
                    # Score nodes by number of assigned neighbors
                    max_cnt = -1
                    best_nodes = []
                    for v in unassigned:
                        cnt = 0
                        for nb in adj[v]:
                            if mapping[nb] != -1:
                                cnt += 1
                        if cnt > max_cnt:
                            max_cnt = cnt
                            best_nodes = [v]
                        elif cnt == max_cnt:
                            best_nodes.append(v)
                    
                    # Randomize choice among best nodes
                    rng.shuffle(best_nodes)
                    best_node = best_nodes[0]
                    
                    # Candidate host vertices
                    host_candidates = []
                    for h in range(n):
                        if host_used[h]:
                            continue
                        valid = True
                        for nb in adj[best_node]:
                            if mapping[nb] != -1:
                                h_nb = mapping[nb]
                                edge = (min(h, h_nb), max(h, h_nb))
                                if edge in used_edges:
                                    valid = False
                                    break
                        if valid:
                            host_candidates.append(h)
                            
                    rng.shuffle(host_candidates)
                    
                    for h in host_candidates:
                        mapping[best_node] = h
                        host_used[h] = True
                        if dfs(depth + 1):
                            return True
                        mapping[best_node] = -1
                        host_used[h] = False
                        
                    return False
                
                if dfs(0):
                    result[idx] = mapping[:]
                    for u, v in trees[idx]:
                        hu = mapping[u]
                        hv = mapping[v]
                        edge = (min(hu, hv), max(hu, hv))
                        used_edges.add(edge)
                else:
                    success = False
                    break
                    
            if success:
                value = sum(len(trees[i]) for i in range(n) if result[i] is not None)
                if value > best_value:
                    best_value = value
                    best_result = result[:]
                    if value == 190:
                        return [ [int(x) for x in r] if r is not None else None for r in result]
                        
    # Fallback: Return best partial solution
    if best_value > 0:
        return [ [int(x) for x in r] if r is not None else None for r in best_result]
        
    # If all failed, try to embed just T20
    idx = 19
    k = 20
    adj = tree_adj[idx]
    mapping = [-1] * k
    host_used = [False] * n
    rng = Random(42)
    
    def dfs_fallback(depth):
        if depth == k:
            return True
        unassigned = [v for v in range(k) if mapping[v] == -1]
        if not unassigned:
            return False
        max_cnt = -1
        best_nodes = []
        for v in unassigned:
            cnt = 0
            for nb in adj[v]:
                if mapping[nb] != -1:
                    cnt += 1
            if cnt > max_cnt:
                max_cnt = cnt
                best_nodes = [v]
            elif cnt == max_cnt:
                best_nodes.append(v)
        rng.shuffle(best_nodes)
        best_node = best_nodes[0]
        
        host_candidates = [h for h in range(n) if not host_used[h]]
        rng.shuffle(host_candidates)
        
        for h in host_candidates:
            mapping[best_node] = h
            host_used[h] = True
            if dfs_fallback(depth + 1):
                return True
            mapping[best_node] = -1
            host_used[h] = False
        return False
        
    if dfs_fallback(0):
        best_result[idx] = mapping[:]
        best_value = len(trees[idx])
        return [ [int(x) for x in r] if r is not None else None for r in best_result]
        
    return [None] * n
