<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best_adj = None
    best_edges = 0
    
    # Start with Mycielski of C5 (Grötzsch graph) + 5 isolated vertices
    adj_base = _mycielski_of_c5()
    n = 16
    adj = [set(s) for s in adj_base] + [set() for _ in range(5)]
    
    # First, try to connect isolated vertices to existing structure
    # Each new vertex v can connect to an independent set in existing graph
    for new_v in range(11, 16):
        # Find maximum independent set in current graph that new_v can connect to
        # Actually, new_v can connect to a set S where S is independent in G
        # and no vertex in S is adjacent to any other vertex in S (trivially true for independent set)
        # The constraint is: if new_v connects to S, then for any edge (u,w) in G, not both u,w in S
        # which is exactly that S is independent.
        # To maximize edges, connect new_v to a maximum independent set.
        
        # Simple heuristic: try to add edges one by one
        for u in range(new_v):
            if time.time() > deadline:
                break
            # Check if adding (new_v, u) keeps triangle-free
            can_add = True
            for w in adj[new_v]:
                if w in adj[u]:
                    can_add = False
                    break
            if can_add:
                adj[new_v].add(u)
                adj[u].add(new_v)
    
    # Now do local search to add more edges
    best_adj, best_edges = _hill_climbing_from(adj, deadline, rng)
    
    # Try multiple random restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Generate a random triangle-free graph
        adj_rand = _random_triangle_free_graph(16, seed=rng.randint(0, 10**6))
        
        # Check if non-3-colorable
        if not _is_not_3_colorable(adj_rand):
            continue
        
        # Try to improve
        adj_improved, edges_improved = _hill_climbing_from(adj_rand, deadline - 2, rng)
        
        if edges_improved > best_edges:
            best_edges = edges_improved
            best_adj = adj_improved
    
    # Also try starting from Grötzsch + various connections
    if time.time() < deadline - 5:
        for trial in range(20):
            if time.time() > deadline - 2:
                break
            adj_try = _mycielski_of_c5()
            adj_try = [set(s) for s in adj_try] + [set() for _ in range(5)]
            # Randomly connect new vertices
            for new_v in range(11, 16):
                candidates = list(range(new_v))
                rng.shuffle(candidates)
                for u in candidates:
                    if time.time() > deadline - 2:
                        break
                    can_add = True
                    for w in adj_try[new_v]:
                        if w in adj[u]:
                            can_add = False
                            break
                    if can_add:
                        adj_try[new_v].add(u)
                        adj_try[u].add(new_v)
            
            if _is_not_3_colorable(adj_try):
                adj_imp, ed = _hill_climbing_from(adj_try, deadline - 2, rng)
                if ed > best_edges:
                    best_edges = ed
                    best_adj = adj_imp
    
    # Final result
    if best_adj is None:
        # Fallback: return Grötzsch graph with isolated vertices
        adj = _mycielski_of_c5()
        adj = [set(s) for s in adj] + [set() for _ in range(5)]
        best_adj = adj
    
    # Convert to edge list
    edges = []
    for u in range(16):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
=======
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best_adj = None
    best_edges = 0
    
    # Start with Mycielski of C5 (Grötzsch graph) + 5 isolated vertices
    adj_base = _mycielski_of_c5()
    n = 16
    adj = [set(s) for s in adj_base] + [set() for _ in range(5)]
    
    # First, try to connect isolated vertices to existing structure
    # Each new vertex v can connect to an independent set in existing graph
    for new_v in range(11, 16):
        # Simple heuristic: try to add edges one by one
        for u in range(new_v):
            if time.time() > deadline:
                break
            # Check if adding (new_v, u) keeps triangle-free
            can_add = True
            for w in adj[new_v]:
                if w in adj[u]:
                    can_add = False
                    break
            if can_add:
                adj[new_v].add(u)
                adj[u].add(new_v)
    
    # Now do local search to add more edges
    best_adj, best_edges = _hill_climbing_from(adj, deadline, rng)
    
    # Try multiple random restarts
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Generate a random triangle-free graph
        adj_rand = _random_triangle_free_graph(16, seed=rng.randint(0, 10**6))
        
        # Check if non-3-colorable
        if not _is_not_3_colorable(adj_rand):
            continue
        
        # Try to improve
        adj_improved, edges_improved = _hill_climbing_from(adj_rand, deadline - 5, rng)
        
        if edges_improved > best_edges:
            best_edges = edges_improved
            best_adj = adj_improved
    
    # Use simulated annealing on the best result found so far
    if best_adj is not None and time.time() < deadline - 5:
        sa_adj, sa_edges = _simulated_annealing(best_adj, deadline - 2, rng)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj
    
    # Also try starting from Grötzsch + various connections
    if time.time() < deadline - 5:
        for trial in range(10):
            if time.time() > deadline - 2:
                break
            adj_try = _mycielski_of_c5()
            adj_try = [set(s) for s in adj_try] + [set() for _ in range(5)]
            # Randomly connect new vertices
            for new_v in range(11, 16):
                candidates = list(range(new_v))
                rng.shuffle(candidates)
                for u in candidates:
                    if time.time() > deadline - 2:
                        break
                    can_add = True
                    for w in adj_try[new_v]:
                        if w in adj[u]:
                            can_add = False
                            break
                    if can_add:
                        adj_try[new_v].add(u)
                        adj_try[u].add(new_v)
            
            if _is_not_3_colorable(adj_try):
                adj_imp, ed = _hill_climbing_from(adj_try, deadline - 2, rng)
                if ed > best_edges:
                    best_edges = ed
                    best_adj = adj_imp
    
    # Final simulated annealing polish
    if best_adj is not None and time.time() < deadline - 1:
        sa_adj, sa_edges = _simulated_annealing(best_adj, deadline - 0.5, rng)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj
    
    # Final result
    if best_adj is None:
        # Fallback: return Grötzsch graph with isolated vertices
        adj = _mycielski_of_c5()
        adj = [set(s) for s in adj] + [set() for _ in range(5)]
        best_adj = adj
    
    # Convert to edge list
    edges = []
    for u in range(16):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
>>>>>>> REPLACE
