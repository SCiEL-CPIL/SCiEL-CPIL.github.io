import itertools

def _generate_trees(n=20, seed=611):
    from random import Random
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

def _build_adjacency(n, used_edges):
    adj = [set() for _ in range(n)]
    for a, b in used_edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _find_embedding(tree, k, n, used_edges):
    """
    Find an embedding of tree (list of (u,v) edges on vertices 0..k-1) into K_n
    such that no edge conflicts with used_edges.
    Returns a list of length k where result[i] is the host vertex for tree vertex i,
    or None if not found.
    """
    if k == 1:
        return [0]
    if k == 2:
        for a in range(n):
            for b in range(a + 1, n):
                if (a, b) not in used_edges:
                    return [a, b]
        return None

    # Build tree adjacency
    adj = [[] for _ in range(k)]
    for u, v in tree:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a leaf to start from
    leaf = None
    for i in range(k):
        if len(adj[i]) == 1:
            leaf = i
            break
    if leaf is None:
        leaf = 0

    # Use backtracking with constraint propagation
    # Order: start from leaf, expand
    parent = [-1] * k
    parent[leaf] = -2  # mark as root
    order = [leaf]
    queue = [leaf]
    idx = 0
    while idx < len(queue):
        u = queue[idx]
        idx += 1
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                order.append(v)
                queue.append(v)
    
    # order is a BFS ordering starting from leaf
    # Now backtrack: assign host vertices in order
    
    # Precompute for each tree vertex, which tree vertices are already assigned
    # and their host vertices, to check edge conflicts
    
    host = [-1] * k
    
    def backtrack(pos):
        if pos == len(order):
            return True
        
        v = order[pos]
        # Determine which neighbors of v in the tree are already assigned
        # and thus constrain the host vertex for v
        parent_v = parent[v]
        # Children of v in the BFS tree are those w where parent[w] == v
        # But we also need to check all tree neighbors, not just BFS tree neighbors
        
        # For each candidate host vertex h in 0..n-1:
        # - h must not be used by any other tree vertex
        # - for each tree neighbor u of v that is already assigned, (host[u], h) must not be in used_edges
        # - h must not create conflict with used edges
        
        # Get already-assigned neighbors of v in the tree
        assigned_neighbors = []
        for u in adj[v]:
            if host[u] != -1:
                assigned_neighbors.append(u)
        
        # Track which host vertices are already used
        used_hosts = set()
        for i in range(k):
            if host[i] != -1:
                used_hosts.add(host[i])
        
        for h in range(n):
            if h in used_hosts:
                continue
            # Check edge conflicts with already-assigned neighbors
            ok = True
            for u in assigned_neighbors:
                hu = host[u]
                edge = (min(h, hu), max(h, hu))
                if edge in used_edges:
                    ok = False
                    break
            if not ok:
                continue
            
            host[v] = h
            if backtrack(pos + 1):
                return True
            host[v] = -1
        
        return False
    
    # Initialize: assign the root
    root = order[0]
    for h in range(n):
        host[root] = h
        if backtrack(1):
            return [host[i] for i in range(k)]
        host[root] = -1
    
    return None

def construct():
    n = 20
    trees = _generate_trees(n)
    
    result = [None] * n
    used_edges = set()
    
    # Embed trees in order of decreasing size for better packing
    # T20 has 19 edges, T19 has 18, ..., T1 has 0
    tree_indices = list(range(n - 1, -1, -1))  # 19, 18, ..., 0 (0-indexed)
    
    for idx in tree_indices:
        k = idx + 1
        tree = trees[idx]
        
        if k == 1:
            result[idx] = None
            continue
        
        embedding = _find_embedding(tree, k, n, used_edges)
        if embedding is None:
            # This shouldn't happen for small trees, but if it does, return what we have
            return result
        
        # Convert to list of Python ints
        embedding = [int(x) for x in embedding]
        result[idx] = embedding
        
        # Add edges to used_edges
        for u, v in tree:
            hu = embedding[u]
            hv = embedding[v]
            edge = (min(hu, hv), max(hu, hv))
            used_edges.add(edge)
    
    return result
