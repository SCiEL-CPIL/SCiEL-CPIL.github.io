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
    deadline = start_time + 115.0  # safety margin below 120s limit
    
    N = 20
    trees = instance_trees(N, 611)
    
    # Precompute tree structures
    tree_info = []
    for idx in range(N):
        edges = trees[idx]
        k = idx + 1
        if k <= 2:
            tree_info.append((k, edges, []))
            continue
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
            deg[a] += 1
            deg[b] += 1
        # MRV order: sort by degree descending (most constrained first)
        var_order = sorted(range(k), key=lambda v: -deg[v])
        tree_info.append((k, edges, var_order))
    
    def embed_tree(tree_idx, used_edges, rng, node_limit=200000):
        """
        Try to embed tree 'tree_idx' into the remaining available edges.
        Returns a list of host vertices for the tree vertices, or None.
        """
        k, edges, var_order = tree_info[tree_idx]
        
        if k == 1:
            # Any unused vertex
            for h in range(N):
                if h not in used_edges:  # Actually, used_edges is a set of edges, not vertices
                    return [h]
            return None
            
        if k == 2:
            # Find an unused edge
            candidates = []
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) not in used_edges and (v, u) not in used_edges:
                        candidates.append((u, v))
            if not candidates:
                return None
            u, v = rng.choice(candidates)
            return [u, v]
        
        # Build adjacency
        adj = [[] for _ in range(k)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
            
        # Compute host remaining degrees
        host_deg = [0] * N
        for (x, y) in used_edges:
            host_deg[x] += 1
            host_deg[y] += 1
        host_rem = [N - 1 - host_deg[h] for h in range(N)]
        
        mapping = [-1] * k
        used_hosts = set()
        nodes = [0]
        
        def backtrack(idx):
            if nodes[0] > node_limit or time.time() > deadline:
                return False
            if idx == k:
                return True
                
            v = var_order[idx]
            cands = []
            for h in range(N):
                if h in used_hosts:
                    continue
                ok = True
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        e1 = (min(h, mw), max(h, mw))
                        if e1 in used_edges:
                            ok = False
                            break
                if ok:
                    cands.append(h)
                    
            if not cands:
                return False
                
            # Heuristic: for high-degree tree vertices, prefer host vertices with high remaining degree
            deg_v = len(adj[v])
            if deg_v >= 3:
                cands.sort(key=lambda h: -host_rem[h])
            else:
                rng.shuffle(cands)
                
            for h in cands:
                nodes[0] += 1
                if nodes[0] > node_limit or time.time() > deadline:
                    return False
                    
                mapping[v] = h
                used_hosts.add(h)
                added = []
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        e = (min(h, mw), max(h, mw))
                        used_edges.add(e)
                        added.append(e)
                        
                if backtrack(idx + 1):
                    return True
                    
                for e in added:
                    used_edges.discard(e)
                used_hosts.discard(h)
                mapping[v] = -1
                
            return False
            
        if backtrack(0):
            return [int(mapping[v]) for v in range(k)]
        return None
    
    def evaluate(order, rng, node_limit=200000):
        used_edges = set()
        result = [None] * N
        total = 0
        for idx in order:
            if time.time() > deadline:
                break
            k, edges, _ = tree_info[idx]
            
            if k == 1:
                # Find any unused vertex
                for h in range(N):
                    result[idx] = [h]
                    total += 0
                    break
                continue
                
            if k == 2:
                candidates = []
                for u in range(N):
                    for v in range(u + 1, N):
                        if (u, v) not in used_edges:
                            candidates.append((u, v))
                if not candidates:
                    break
                u, v = rng.choice(candidates)
                result[idx] = [u, v]
                used_edges.add((u, v))
                total += 1
                continue
            
            embedding = embed_tree(idx, used_edges, rng, node_limit)
            if embedding is not None:
                result[idx] = embedding
                for (a, b) in edges:
                    u, v = embedding[a], embedding[b]
                    e = (min(u, v), max(u, v))
                    used_edges.add(e)
                total += len(edges)
            else:
                break
                
        return result, total
    
    # Try multiple orderings with restarts
    best_result = None
    best_val = -1
    
    # Strategy 1: Large-first (most constrained first)
    base_order = list(range(N - 1, -1, -1))
    
    master_rng = Random(12345)
    
    # First try the base large-first order
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        rng = Random(attempt * 7 + 1)
        result, val = evaluate(base_order, rng, node_limit=300000)
        if val > best_val:
            best_val = val
            best_result = result
        if best_val == 190:
            return best_result
    
    # Strategy 2: Large-first with shuffled tail
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        order = list(range(N - 1, -1, -1))
        # Keep largest first, shuffle the rest
        master_rng.shuffle(order[1:])
        rng = Random(attempt * 13 + 2)
        result, val = evaluate(order, rng, node_limit=200000)
        if val > best_val:
            best_val = val
            best_result = result
        if best_val == 190:
            return best_result
    
    # Strategy 3: Random orderings
    for attempt in range(300):
        if time.time() > deadline - 3:
            break
        order = list(range(N))
        master_rng.shuffle(order)
        rng = Random(attempt * 17 + 3)
        result, val = evaluate(order, rng, node_limit=150000)
        if val > best_val:
            best_val = val
            best_result = result
        if best_val == 190:
            return best_result
    
    # Strategy 4: Degree-based ordering (trees with high max degree first)
    max_degrees = []
    for idx in range(N):
        k, edges, _ = tree_info[idx]
        if k <= 2:
            max_degrees.append((0, idx))
        else:
            deg = [0] * k
            for a, b in edges:
                deg[a] += 1
                deg[b] += 1
            max_degrees.append((max(deg), idx))
    max_degrees.sort(key=lambda x: -x[0])
    deg_order = [idx for _, idx in max_degrees]
    
    for attempt in range(50):
        if time.time() > deadline - 2:
            break
        rng = Random(attempt * 23 + 4)
        result, val = evaluate(deg_order, rng, node_limit=200000)
        if val > best_val:
            best_val = val
            best_result = result
        if best_val == 190:
            return best_result
    
    if best_result is not None and best_val > 0:
        # Ensure all entries are proper Python lists of ints or None
        final = []
        for r in best_result:
            if r is None:
                final.append(None)
            else:
                final.append([int(x) for x in r])
        return final
    
    # Fallback: just embed T20
    result = [None] * N
    result[19] = list(range(20))
    return result
