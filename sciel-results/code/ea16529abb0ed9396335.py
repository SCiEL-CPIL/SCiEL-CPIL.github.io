import time
import random
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
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency and structure for each tree
    tree_data = []
    for t_idx, t in enumerate(trees):
        k = t_idx + 1
        if k == 1:
            tree_data.append((k, [], []))
            continue
            
        adj = [[] for _ in range(k)]
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
        
        # Compute subtree sizes and ordered children
        # Root at 0
        parent = [-1] * k
        order = []
        stack = [0]
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if v != parent[u]:
                    parent[v] = u
                    stack.append(v)
        
        size = [1] * k
        children = [[] for _ in range(k)]
        for u in reversed(order):
            for v in adj[u]:
                if v != parent[u]:
                    children[u].append(v)
                    size[u] += size[v]
        
        # Sort children by size descending to prioritize larger subtrees
        for u in range(k):
            children[u].sort(key=lambda x: size[x], reverse=True)
            
        tree_data.append((k, children, size))
    
    # Sort trees by size descending for packing order
    order = list(range(n))
    order.sort(key=lambda x: len(trees[x]), reverse=True)
    
    best_solution = None
    best_value = 0
    
    # Try multiple random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 12345 + 678)
        
        # Initialize available neighbors for K20
        # neighbors[i] is a set of vertices j such that (i,j) is available
        neighbors = [set(range(n)) for _ in range(n)]
        for i in range(n):
            neighbors[i].remove(i)
            
        current_solution = [None] * n
        success = True
        packed_edges = 0
        
        for t_idx in order:
            k, children, size = tree_data[t_idx]
            
            if k == 1:
                # T1: Single vertex
                v = rng.randint(0, n - 1)
                current_solution[t_idx] = [v]
                continue
            
            # Try to embed
            mapping = [-1] * k
            used_edges = []
            
            def embed(u, parent_gv):
                if time.time() > deadline:
                    return False
                
                if parent_gv == -1:
                    # Root: choose any unused vertex
                    # Heuristic: pick a vertex with high degree in available graph? 
                    # Or just random. Random is fine with restarts.
                    candidates = list(range(n))
                    rng.shuffle(candidates)
                else:
                    # Children: must be adjacent to parent in K20 and edge available
                    candidates = list(neighbors[parent_gv])
                    rng.shuffle(candidates)
                
                for gv in candidates:
                    if gv in mapping:
                        continue
                    
                    mapping[u] = gv
                    ok = True
                    
                    # Recurse on children
                    for child in children[u]:
                        if not embed(child, gv):
                            ok = False
                            break
                    
                    if ok:
                        return True
                    
                    mapping[u] = -1
                
                return False
            
            # We need to track used edges to remove them from 'neighbors'
            # The recursive function above doesn't update 'neighbors'.
            # We need a version that does, or we track them and update after success.
            # Since backtracking requires undo, it's better to update 'neighbors' inside.
            
            # Let's rewrite embed to update neighbors
            def embed_with_update(u, parent_gv):
                if time.time() > deadline:
                    return False
                
                if parent_gv == -1:
                    candidates = list(range(n))
                    rng.shuffle(candidates)
                else:
                    candidates = list(neighbors[parent_gv])
                    rng.shuffle(candidates)
                
                for gv in candidates:
                    if gv in mapping:
                        continue
                    
                    mapping[u] = gv
                    ok = True
                    edge_to_remove = None
                    if parent_gv != -1:
                        edge_to_remove = (min(parent_gv, gv), max(parent_gv, gv))
                        # Remove edge from neighbors
                        neighbors[parent_gv].remove(gv)
                        neighbors[gv].remove(parent_gv)
                    
                    for child in children[u]:
                        if not embed_with_update(child, gv):
                            ok = False
                            break
                    
                    if ok:
                        return True
                    
                    # Undo
                    if parent_gv != -1:
                        neighbors[parent_gv].add(gv)
                        neighbors[gv].add(parent_gv)
                    mapping[u] = -1
                
                return False
            
            if not embed_with_update(0, -1):
                success = False
                break
            
            # Count edges
            packed_edges += k - 1
            current_solution[t_idx] = mapping[:]
        
        if success:
            # Found a complete packing
            return current_solution
    
    # Fallback
    result = [None] * n
    result[0] = [0]
    result[1] = [0, 1]
    return result

# Ensure Random is available
from random import Random
