import time
from random import Random

def _instance_trees(n=20, seed=611):
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

def _build_adj(tree_edges, k):
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _embed_tree(tree_edges, k, used_edges, rng, deadline):
    """
    Embed a tree into K20 using unused edges.
    Returns a list of host vertices or None.
    """
    if k == 1:
        # Pick any available vertex
        for v in range(20):
            return [v]
        return None

    if k == 2:
        for u in range(20):
            for v in range(u + 1, 20):
                if (u, v) not in used_edges and (v, u) not in used_edges:
                    return [u, v]
        return None

    adj = _build_adj(tree_edges, k)
    
    # We use a DFS backtracking approach.
    # To improve performance, we order the tree vertices in a DFS order from a root.
    # We try different roots if the first one fails.
    
    # Precompute neighbors for each tree vertex
    tree_neighbors = [[] for _ in range(k)]
    for u, v in tree_edges:
        tree_neighbors[u].append(v)
        tree_neighbors[v].append(u)

    # Try to embed with different roots
    # Randomize the order of roots to avoid getting stuck in a bad local minimum
    roots = list(range(k))
    rng.shuffle(roots)
    
    for root in roots:
        if time.time() > deadline:
            return None
            
        # Generate a DFS order from this root
        visited = [False] * k
        order = []
        parent = [-1] * k
        stack = [root]
        visited[root] = True
        while stack:
            cur = stack.pop()
            order.append(cur)
            for nb in tree_neighbors[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = cur
                    stack.append(nb)
        
        assignment = [-1] * k
        available = [True] * 20

        def dfs(idx):
            if time.time() > deadline:
                return False
                
            if idx == len(order):
                return True
                
            tv = order[idx]
            
            if tv == root:
                # Try all available host vertices
                hosts = [h for h in range(20) if available[h]]
                rng.shuffle(hosts)
                for h in hosts:
                    assignment[tv] = h
                    available[h] = False
                    if dfs(idx + 1):
                        return True
                    available[h] = True
                    assignment[tv] = -1
                return False
            else:
                p = parent[tv]
                ph = assignment[p]
                if ph == -1:
                    return False
                
                # Try all available host vertices adjacent to ph with unused edge
                hosts = []
                for h in range(20):
                    if available[h] and h != ph:
                        if (ph, h) not in used_edges and (h, ph) not in used_edges:
                            hosts.append(h)
                
                # Heuristic: sort hosts by degree in the remaining graph? 
                # For now, random shuffle is sufficient given the small size.
                rng.shuffle(hosts)
                
                for h in hosts:
                    assignment[tv] = h
                    available[h] = False
                    if dfs(idx + 1):
                        return True
                    available[h] = True
                    assignment[tv] = -1
                return False

        if dfs(0):
            # Verify edge disjointness (safety check)
            for u, v in tree_edges:
                hu, hv = assignment[u], assignment[v]
                if (hu, hv) in used_edges or (hv, hu) in used_edges:
                # This shouldn't happen if dfs logic is correct, but just in case
                    return None
            return [assignment[i] for i in range(k)]
            
    return None

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We use 140s as deadline for safety.
    deadline = start_time + 140.0
    
    trees = _instance_trees(20, 611)
    
    # We want to maximize the number of embedded trees.
    # The optimal solution embeds all 20 trees.
    # Strategy: Embed trees in descending order of size (k=20 down to 1).
    # This is often more effective for constraint satisfaction.
    
    # We will try multiple randomized trials to ensure we find a solution.
    # Since the instance is fixed, a good heuristic should find it quickly.
    
    best_result = [None] * 20
    best_value = 0
    
    # Try a few different random seeds/orders if the first fails, 
    # but descending order with random root selection is usually robust.
    
    for trial in range(100):
        if time.time() > deadline:
            break
            
        rng = Random(trial * 12345 + 67890)
        used_edges = set()
        result = [None] * 20
        total_edges = 0
        success = True
        
        # Descending order: T20, T19, ..., T1
        order = list(range(20, 0, -1))
        
        for k in order:
            if time.time() > deadline:
                success = False
                break
                
            tree_edges = trees[k - 1]
            
            if not tree_edges:
                # T1: single vertex
                # Pick a random available vertex
                avail = [v for v in range(20)]
                # T1 doesn't consume edges, so any vertex is fine.
                # We just need to ensure we return a valid int.
                result[k - 1] = [rng.choice(avail)]
                continue
            
            embedding = _embed_tree(tree_edges, k, used_edges, rng, deadline)
            
            if embedding is None:
                success = False
                break
            
            result[k - 1] = embedding
            for u, v in tree_edges:
                hu, hv = embedding[u], embedding[v]
                used_edges.add((hu, hv))
                used_edges.add((hv, hu))
            total_edges += len(tree_edges)
            
        if success and total_edges > best_value:
            best_value = total_edges
            best_result = result[:]
            if best_value == 190:
                break

    # Ensure types are correct
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
            
    return final
