import time
import random

def instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            trees.append([] if k == 1 else [(0, 1)])
            continue
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
    start_time = time.time()
    timeout = 145.0
    
    trees = instance_trees(20, 611)
    
    # Sort trees by size (descending)
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Pre-compute vertex order for each tree (high degree first)
    tree_vertex_orders = []
    for i in tree_indices:
        edges = trees[i]
        k = i + 1
        if k == 1:
            tree_vertex_orders.append([0])
            continue
        deg = [0] * k
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        verts = list(range(k))
        verts.sort(key=lambda v: deg[v], reverse=True)
        tree_vertex_orders.append(verts)
        
    best_score = 0
    best_solution = None
    rng = random.Random(611)
    
    for restart in range(10000):
        if time.time() - start_time > timeout:
            break
            
        used_edges = set()
        maps = [None] * 20
        
        def backtrack(t_idx):
            if t_idx == len(tree_indices):
                return True
            
            tree_idx = tree_indices[t_idx]
            k = tree_idx + 1
            edges = trees[tree_idx]
            order = tree_vertex_orders[t_idx]
            assign = [-1] * k
            
            if t_idx == 0:
                assign[order[0]] = 0
                if backtrack_assign(t_idx, order, 1, assign, maps, edges):
                    return True
            else:
                if backtrack_assign(t_idx, order, 0, assign, maps, edges):
                    return True
            return False

        def backtrack_assign(t_idx, order, order_idx, assign, maps, edges):
            if order_idx == len(order):
                for u, v in edges:
                    hu = assign[u]
                    hv = assign[v]
                    if hu > hv: hu, hv = hv, hu
                    used_edges.add((hu, hv))
                
                maps[tree_idx] = assign[:]
                
                if backtrack(t_idx + 1):
                    return True
                
                for u, v in edges:
                    hu = assign[u]
                    hv = assign[v]
                    if hu > hv: hu, hv = hv, hu
                    used_edges.remove((hu, hv))
                maps[tree_idx] = None
                return False

            v = order[order_idx]
            hosts = list(range(20))
            rng.shuffle(hosts)
            
            for h in hosts:
                if h in assign:
                    continue
                
                valid = True
                for u, w in edges:
                    if u == v:
                        other = w
                    elif w == v:
                        other = u
                    else:
                        continue
                    
                    if assign[other] != -1:
                        ho = assign[other]
                        if h > ho:
                            h, ho = ho, h
                        if (h, ho) in used_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                assign[v] = h
                if backtrack_assign(t_idx, order, order_idx + 1, assign, maps, edges):
                    return True
                assign[v] = -1
                
            return False

        if backtrack(0):
            res = []
            for i in range(20):
                if maps[i] is None:
                    res.append(None)
                else:
                    res.append([int(x) for x in maps[i]])
            
            edge_count = sum(len(trees[i]) for i in range(20) if maps[i] is not None)
            
            if edge_count > best_score:
                best_score = edge_count
                best_solution = res
                
            if best_score == 190:
                break

    if best_solution is None:
        return [None] * 19 + [list(range(20))]
        
    return best_solution
