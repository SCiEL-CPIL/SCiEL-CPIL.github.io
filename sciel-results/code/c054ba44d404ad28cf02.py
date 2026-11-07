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
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency and degree for each tree
    tree_adj = []
    tree_deg = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_deg.append(deg)
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Sort trees by size descending for packing order
    order = list(range(n))
    order.sort(key=lambda x: len(trees[x]), reverse=True)
    
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 7919 + 13)
        
        available = set(all_edges)
        current_solution = [None] * n
        success = True
        packed_edges = 0
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                # T1: Single vertex, no edges
                v = rng.randint(0, n - 1)
                current_solution[t_idx] = [v]
                continue
            
            adj = tree_adj[t_idx]
            deg = tree_deg[t_idx]
            
            # Pick root: prefer high-degree nodes for better embedding
            if rng.random() < 0.8:
                max_deg = max(deg)
                high_deg_nodes = [i for i in range(k) if deg[i] == max_deg]
                root = rng.choice(high_deg_nodes)
            else:
                root = rng.randint(0, k - 1)
            
            mapping = [-1] * k
            
            def embed_dfs(node, parent):
                if time.time() > deadline:
                    return False
                
                if parent != -1:
                    parent_gv = mapping[parent]
                    # Find available neighbors of parent_gv
                    candidates = []
                    for v in range(n):
                        if v == parent_gv:
                            continue
                        e = (min(parent_gv, v), max(parent_gv, v))
                        if e in available:
                            candidates.append(v)
                else:
                    candidates = list(range(n))
                
                # Shuffle for randomization
                rng.shuffle(candidates)
                
                for gv in candidates:
                    if gv in mapping:
                        continue
                    
                    mapping[node] = gv
                    ok = True
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if not embed_dfs(neighbor, node):
                            ok = False
                            break
                    
                    if ok:
                        return True
                    
                    mapping[node] = -1
                
                return False
            
            if not embed_dfs(root, -1):
                success = False
                break
            
            # Update available edges after successful embedding
            for u, v in trees[t_idx]:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                if e in available:
                    available.remove(e)
                else:
                    # Should not happen, but safety check
                    success = False
                    break
                packed_edges += 1
            
            if not success:
                break
            
            current_solution[t_idx] = mapping[:]
        
        if success:
            # Found a complete packing - return immediately
            # Ensure all entries are proper lists of Python ints
            result = []
            for i in range(n):
                if current_solution[i] is None:
                    result.append(None)
                else:
                    result.append([int(x) for x in current_solution[i]])
            return result
    
    # Fallback: return a minimal valid solution
    result = [None] * n
    result[0] = [0]
    result[1] = [0, 1]
    return result
