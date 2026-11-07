def construct():
    """
    Searches for the maximum-edge valid graph on 15 vertices where
    no 3-path has endpoints of equal degree.
    """
    import time
    
    n = 15
    deadline = time.time() + 80
    
    # All possible edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    # Start with K_{7,8} as initial solution (56 edges)
    init_edges = set()
    for u in range(7):
        for v in range(7, 15):
            init_edges.add((u, v))
    
    def compute_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edge_set, deg):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Verify initial
    deg_init = compute_degrees(init_edges)
    assert is_valid(init_edges, deg_init)
    
    best_count = len(init_edges)
    best_set = set(init_edges)
    
    # Try to add edges one by one
    current_edges = set(init_edges)
    current_deg = compute_degrees(current_edges)
    
    # Remaining edges not in current
    remaining = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in current_edges]
    
    # Greedy: try adding each remaining edge
    for edge in remaining:
        if time.time() > deadline:
            break
        if edge in current_edges:
            continue
        new_set = set(current_edges)
        new_set.add(edge)
        new_deg = compute_degrees(new_set)
        if is_valid(new_set, new_deg):
            current_edges = new_set
            current_deg = new_deg
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_set = set(current_edges)
    
    # Also try: remove some edges from K_{7,8} and add many others
    # This is a local search approach
    # Try random restarts with different initial structures
    
    # Try a few different partition sizes
    for a_size in range(5, 11):
        if time.time() > deadline:
            break
        b_size = n - a_size
        if a_size > b_size:
            continue
        test_edges = set()
        for u in range(a_size):
            for v in range(a_size, n):
                test_edges.add((u, v))
        test_deg = compute_degrees(test_edges)
        if is_valid(test_edges, test_deg):
            # Try to add more edges
            cur = set(test_edges)
            cur_deg = test_deg
            rem = [(i,j) for i in range(n) for j in range(i+1,n) if (i,j) not in cur]
            for e in rem:
                if time.time() > deadline:
                    break
                if e in cur:
                    continue
                ns = set(cur)
                ns.add(e)
                nd = compute_degrees(ns)
                if is_valid(ns, nd):
                    cur = ns
                    cur_deg = nd
            if len(cur) > best_count:
                best_count = len(cur)
                best_set = set(cur)
    
    result = [[u, v] for u, v in best_set]
    return result
