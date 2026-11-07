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
    deadline = start_time + 140
    
    trees = instance_trees(20, 611)
    
    # Precompute adjacency for each tree
    tree_adj = []
    for i, tree in enumerate(trees):
        k = i + 1
        adj = [[] for _ in range(k)]
        for a, b in tree:
            adj[a].append(b)
            adj[b].append(a)
        tree_adj.append(adj)
    
    # Degree of each vertex in each tree
    tree_deg = []
    for i, adj in enumerate(tree_adj):
        tree_deg.append([len(adj[v]) for v in range(len(adj))])
    
    best_result = [None] * 20
    best_edges = 0
    
    def count_edges(result):
        total = 0
        for i in range(20):
            if result[i] is not None:
                total += max(0, i)
        return total
    
    # Use backtracking with good heuristics
    # Process trees in order, for each tree do a DFS with pruning
    
    def embed_tree(tree_idx, used_edges, rng, time_limit):
        """
        Try to embed tree tree_idx into K20 using only edges not in used_edges.
        Returns assignment list or None.
        """
        k = tree_idx + 1
        if k == 1:
            return [rng.randint(0, 19)]
        
        adj = tree_adj[tree_idx]
        deg = tree_deg[tree_idx]
        
        # Choose root: highest degree vertex
        root = max(range(k), key=lambda v: (deg[v], -v))
        
        # Get BFS order
        visited = [False] * k
        order = [root]
        visited[root] = True
        parent = [-1] * k
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        
        # Backtracking
        assign = [-1] * k
        assign[root] = rng.randint(0, 19)
        start_bt = time.time()
        
        def bt(pos):
            if time.time() - start_bt > 2.0:  # 2 second limit per tree
                return False
            if pos == len(order):
                return True
            
            v = order[pos]
            p = parent[v]
            pu = assign[p]
            
            # Get candidate host vertices
            used_hosts = set(assign[:pos])
            
            # Order candidates by degree (higher degree first for pruning)
            candidates = []
            for hu in range(20):
                if hu in used_hosts:
                    continue
                edge = (min(pu, hu), max(pu, hu))
                if edge in used_edges:
                    continue
                # Count how many neighbors of v can be placed
                candidates.append(hu)
            
            # Shuffle for randomization
            rng.shuffle(candidates)
            
            for hu in candidates:
                assign[v] = hu
                if bt(pos + 1):
                    return True
                assign[v] = -1
            
            return False
        
        if bt(1):
            return assign[:]
        
        return None
    
    def try_packing(order, rng):
        used_edges = set()
        result = [None] * 20
        start = time.time()
        
        for tree_idx in order:
            if time.time() - start > 130:
                return None
            
            assign = embed_tree(tree_idx, used_edges, rng, 2.0)
            if assign is None:
                return None
            
            # Verify and commit
            tree = trees[tree_idx]
            for a, b in tree:
                ua, ub = assign[a], assign[b]
                edge = (min(ua, ub), max(ua, ub))
                if edge in used_edges:
                    return None
                used_edges.add(edge)
            
            result[tree_idx] = assign[:]
        
        return result
    
    # Try many random orderings
    rng = Random(42)
    order_desc = list(range(19, -1, -1))
    
    for restart in range(10000):
        if time.time() > deadline:
            break
        
        # Create order
        if restart % 3 == 0:
            order = order_desc[:]
        elif restart % 3 == 1:
            order = order_desc[:]
            rng.shuffle(order)
        else:
            order = list(range(20))
            rng.shuffle(order)
        
        res = try_packing(order, rng)
        if res is not None:
            edges = count_edges(res)
            if edges > best_edges:
                best_edges = edges
                best_result = res[:]
                if best_edges == 190:
                    break
    
    final = []
    for i in range(20):
        if best_result[i] is None:
            final.append(None)
        else:
            final.append([int(v) for v in best_result[i]])
    
    return final

