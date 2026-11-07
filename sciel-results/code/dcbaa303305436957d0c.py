import time
from random import Random
from collections import defaultdict

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
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill
    
    trees = instance_trees(20, 611)
    N = 20
    
    # Precompute tree structures
    tree_adj = []
    tree_degrees = []
    tree_edges = []
    for k in range(20):
        edges = trees[k]
        adj = defaultdict(list)
        deg = [0] * (k + 1)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_degrees.append(deg)
        tree_edges.append(edges)
    
    # Total edges in K20 = 190, sum of tree edges = 0+1+...+19 = 190
    # So we must use ALL edges exactly once.
    
    # Embedding order: largest trees first
    embed_order = list(range(19, -1, -1))
    
    # Fast solver using backtracking with edge-disjointness constraints
    def solve_with_order(order):
        used_edges = set()
        embeddings = [None] * 20
        
        for tree_idx in order:
            k = tree_idx + 1
            edges = tree_edges[tree_idx]
            adj = tree_adj[tree_idx]
            deg = tree_degrees[tree_idx]
            
            if k == 1:
                embeddings[tree_idx] = [0]
                continue
            
            # BFS order for this tree, starting from highest degree vertex
            root = max(range(k), key=lambda v: deg[v])
            visited = [False] * k
            t_order = [root]
            visited[root] = True
            queue = [root]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        t_order.append(v)
                        queue.append(v)
            
            # Backtracking with forward checking
            assign = {}  # tree_vertex -> host_vertex
            assigned_hosts = set()
            
            def backtrack(pos):
                if time.time() > deadline:
                    return False
                
                if pos == k:
                    # All assigned, verify edges are not in used_edges and commit
                    for u, v in edges:
                        a, b = assign[u], assign[v]
                        if a > b:
                            a, b = b, a
                        if (a, b) in used_edges:
                            return False
                    for u, v in edges:
                        a, b = assign[u], assign[v]
                        if a > b:
                            a, b = b, a
                        used_edges.add((a, b))
                    embeddings[tree_idx] = [assign[i] for i in range(k)]
                    return True
                
                tv = t_order[pos]
                
                # Generate candidate host vertices, ordered by degree in remaining graph
                candidates = []
                for h in range(N):
                    if h in assigned_hosts:
                        continue
                    # Check against already-assigned neighbors
                    ok = True
                    for nb in adj[tv]:
                        if nb in assign:
                            a, b = h, assign[nb]
                            if a > b:
                                a, b = b, a
                            if (a, b) in used_edges:
                                ok = False
                                break
                    if ok:
                        candidates.append(h)
                
                if not candidates:
                    return False
                
                # Order candidates by how many unused edges they have to unassigned tree neighbors
                # (prefer vertices with more flexibility)
                def score_candidate(h):
                    score = 0
                    for nb in adj[tv]:
                        if nb not in assign:
                            # Count unused edges from h to other unassigned positions
                            for h2 in range(N):
                                if h2 in assigned_hosts or h2 == h:
                                    continue
                                a, b = h, h2
                                if a > b:
                                    a, b = b, a
                                if (a, b) not in used_edges:
                                    score += 1
                    return score
                
                # For performance, only do detailed scoring if candidates are many
                if len(candidates) > 5:
                    candidates.sort(key=score_candidate, reverse=True)
                
                for h in candidates:
                    assign[tv] = h
                    assigned_hosts.add(h)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    del assign[tv]
                    assigned_hosts.remove(h)
                
                return False
            
            if not backtrack(0):
                return None
        
        return embeddings
    
    # Try the main solve
    result = solve_with_order(embed_order)
    if result is not None:
        # Verify
        used = set()
        for i in range(20):
            if result[i] is None:
                return [None] * 20
            for u, v in tree_edges[i]:
                a, b = result[i][u], result[i][v]
                if a > b:
                    a, b = b, a
                if (a, b) in used:
                    return [None] * 20
                used.add((a, b))
        return result
    
    # If main failed, try randomized restarts
    rng = Random(12345)
    for attempt in range(5000):
        if time.time() > deadline - 3:
            break
        
        # Create a perturbed order
        order = embed_order[:]
        for _ in range(rng.randint(1, 5)):
            i = rng.randrange(len(order) - 1)
            order[i], order[i + 1] = order[i + 1], order[i]
        
        result = solve_with_order(order)
        if result is not None:
            used = set()
            for i in range(20):
                if result[i] is None:
                    return [None] * 20
                for u, v in tree_edges[i]:
                    a, b = result[i][u], result[i][v]
                    if a > b:
                        a, b = b, a
                    if (a, b) in used:
                        return [None] * 20
                    used.add((a, b))
            return result
    
    return [None] * 20
