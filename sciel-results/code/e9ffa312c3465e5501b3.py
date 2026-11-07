import time
from random import Random

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
    deadline = time.time() + 110
    trees = instance_trees(20, 611)
    
    # Pre-process trees
    tree_adj = []
    for k in range(1, 21):
        adj = [[] for _ in range(k)]
        for (u, v) in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        
    # Order trees: Largest first
    order = list(range(20, 0, -1))
    
    best_result = [None] * 20
    best_edges = 0
    
    def count_embedded_edges(res):
        total = 0
        for i in range(20):
            if res[i] is not None:
                total += len(trees[i])
        return total
    
    def try_embed_tree(k, adj, used_edges, rng, deadline):
        """Try to embed tree k using backtracking. Returns mapping or None."""
        if k == 1:
            return [rng.randint(0, 19)]
            
        if k == 2:
            for a in range(20):
                for b in range(a+1, 20):
                    edge = (a, b)
                    if edge not in used_edges:
                        return [a, b]
            return None
        
        k_val = k
        deg = [len(adj[i]) for i in range(k_val)]
        v_order = sorted(range(k_val), key=lambda x: -deg[x])
        
        mapping = [-1] * k_val
        host_used = [False] * 20
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == k_val:
                return True
            
            tv = v_order[idx]
            assigned_nbs = [nb for nb in adj[tv] if mapping[nb] != -1]
            
            candidates = []
            for h in range(20):
                if host_used[h]:
                    continue
                ok = True
                for nb in assigned_nbs:
                    hnb = mapping[nb]
                    edge = (min(h, hnb), max(h, hnb))
                    if edge in used_edges:
                        ok = False
                        break
                if ok:
                    candidates.append(h)
                    
            if not candidates:
                return False
            
            rng.shuffle(candidates)
            
            for h in candidates:
                mapping[tv] = h
                host_used[h] = True
                
                if backtrack(idx + 1):
                    return True
                    
                mapping[tv] = -1
                host_used[h] = False
                
            return False
        
        if backtrack(0):
            return [mapping[i] for i in range(k_val)]
        return None
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        rng = Random(attempt * 12345 + 6789)
        
        used_edges = set()
        result = [None] * 20
        failed_at = -1
        
        for k in order:
            if time.time() > deadline:
                failed_at = k
                break
                
            adj = tree_adj[k-1]
            mapping = try_embed_tree(k, adj, used_edges, rng, deadline)
            
            if mapping is None:
                failed_at = k
                break
            
            result[k-1] = mapping
            for (u, v) in trees[k-1]:
                hu, hv = mapping[u], mapping[v]
                used_edges.add((min(hu, hv), max(hu, hv)))
        
        # Record partial solution
        cur_edges = count_embedded_edges(result)
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_result = result[:]
            if best_edges == 190:
                break

    return best_result
