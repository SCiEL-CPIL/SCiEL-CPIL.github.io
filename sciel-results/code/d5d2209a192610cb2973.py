<<<<<<< SEARCH
def _search_voltages(m, g, time_budget):
    """Search for voltages on Petersen edges giving girth >= g.
    Returns voltages list (15 values) or None."""
    pe = _petersen_edges()
    n_edges = 15
    # Candidate voltage sets: try several
    candidates = [
        list(range(1, m)),  # all non-zero
        [1, m//2, 2, m-1] if m >= 4 else [1],
        [1, 2, m-2, m-1] if m >= 4 else [1],
        [1, m//2, (m//2)+1, m-1] if m >= 4 else [1],
    ]
    # Use a deterministic random-like sequence for voltage assignment
    # We'll do a greedy/hill-climbing approach
    # First try: assign voltages to maximize the minimum non-backtracking closed walk voltage
    
    # For efficiency, precompute non-backtracking walks in Petersen up to length g
    # and track their voltage sums for each edge
    # A non-backtracking walk of length L uses L edges. Its total voltage is the sum.
    # We need: for every non-backtracking closed walk of length < g, total voltage != 0 mod m.
    
    # Enumerate non-backtracking closed walks in Petersen up to length g-1
    # Petersen has 10 vertices, 4-regular, 15 edges
    # Non-backtracking walks: state is (current_vertex, previous_vertex)
    # Start from each directed edge, extend without backtracking
    
    # Build adjacency
    adj = [[] for _ in range(10)]
    edge_index = {}  # (u,v) -> edge_idx
    for idx, (u, v) in enumerate(pe):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # Enumerate non-backtracking closed walks of length 2 to g-1
    # A closed non-backtracking walk: v0 -> v1 -> ... -> vL = v0, no immediate backtrack
    # For L < g, we need sum of voltages on edges (v_i, v_{i+1}) != 0 mod m
    
    # Collect all non-backtracking closed walks up to length g-1
    # Represent each as a sequence of edge indices
    closed_walks = []  # list of tuples of edge indices
    
    for L in range(2, g):
        # BFS over states (current, prev) starting from each directed edge
        # Track path of edges
        # For efficiency, just enumerate: from each directed edge (u,v), 
        # extend to (v,w) where w != u, etc.
        # Start states: all directed edges
        for u in range(10):
            for v in adj[u]:
                # Walk: u -> v, edge index
                e0 = edge_index[(u, v)]
                if L == 1:
                    continue
                # Extend to length L and check if closed (ends at u)
                # Use iterative deepening
                # For small L (up to 9) and Petersen (10 nodes), this is feasible
                # State: (current, prev, path_edges)
                # We want walks of exactly length L that end back at u
                # Non-backtracking: next vertex != prev
                
                # DFS
                def dfs(cur, prev, path, depth):
                    if depth == L:
                        if cur == u:
                            closed_walks.append(tuple(path))
                        return
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        e = edge_index[(cur, nxt)]
                        path.append(e)
                        dfs(nxt, cur, path, depth + 1)
                        path.pop()
                
                path = [e0]
                dfs(v, u, path, 1)
    
    # Now we have all non-backtracking closed walks of length 2 to g-1
    # For each, the sum of voltages on its edges must be != 0 mod m
    # This is a constraint satisfaction problem on 15 variables (voltages)
    # with domain [1, m-1] (or restricted)
    
    # Greedy approach: assign voltages one by one, choosing value that satisfies
    # the most constraints
    # Or: try a few structured assignments and verify
    
    # Structured assignment: try to make all length-2 walks have non-zero voltage
    # Length-2 non-backtracking closed walks: u->v->w->u where w != u (i.e., triangle)
    # Petersen has no triangles (girth 5), so no length-3 closed walks
    # Length-4: u->v->w->x->u, non-backtracking
    # Length-5: u->v->w->x->y->u (5-cycle, Petersen has these)
    
    # Simple approach: try random-like assignments with a fixed seed
    # and verify
    import random
    rng = random.Random(42)
    
    best_voltages = None
    for trial in range(200):
        if _elapsed() > time_budget:
            break
        # Choose a candidate set
        cset = candidates[trial % len(candidates)]
        voltages = [rng.choice(cset) for _ in range(n_edges)]
        # Verify: check all closed walks
        ok = True
        for walk in closed_walks:
            total = sum(voltages[e] for e in walk) % m
            if total == 0:
                ok = False
                break
        if ok:
            best_voltages = voltages
            break
        # If not ok, try a perturbation
        if trial > 0 and best_voltages:
            # Perturb from best
            voltages = list(best_voltages)
            for _ in range(3):
                idx = rng.randrange(n_edges)
                voltages[idx] = rng.choice(cset)
            ok = True
            for walk in closed_walks:
                total = sum(voltages[e] for e in walk) % m
                if total == 0:
                    ok = False
                    break
            if ok:
                best_voltages = voltages
                break
    
    return best_voltages
=======
def _search_voltages(m, g, time_budget):
    """Search for voltages on Petersen edges giving girth >= g.
    Returns voltages list (15 values) or None."""
    pe = _petersen_edges()
    n_edges = 15
    
    # Build adjacency and edge index
    adj = [[] for _ in range(10)]
    edge_index = {}
    for idx, (u, v) in enumerate(pe):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx

    # Enumerate non-backtracking closed walks of length 2 to g-1
    closed_walks = []
    for L in range(2, g):
        for u in range(10):
            for v in adj[u]:
                e0 = edge_index[(u, v)]
                if L == 1:
                    continue
                def dfs(cur, prev, path, depth):
                    if depth == L:
                        if cur == u:
                            closed_walks.append(tuple(path))
                        return
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        e = edge_index[(cur, nxt)]
                        path.append(e)
                        dfs(nxt, cur, path, depth + 1)
                        path.pop()
                path = [e0]
                dfs(v, u, path, 1)
    
    # Optimization: Precompute the list of edges for each walk as a list of indices
    # and maybe use numpy for faster sum? No, lists are fine.
    
    import random
    rng = random.Random(42)
    
    best_voltages = None
    # Use full time budget
    start_t = time.time()
    trial = 0
    while time.time() - start_t < time_budget:
        if trial % 1000 == 0:
            # Check global deadline
            if _elapsed() > 130.0:
                break
        # Choose a candidate set
        # Use all non-zero values for better mixing
        voltages = [rng.randint(1, m-1) for _ in range(n_edges)]
        
        # Verify
        ok = True
        for walk in closed_walks:
            total = sum(voltages[e] for e in walk) % m
            if total == 0:
                ok = False
                break
        if ok:
            best_voltages = voltages
            break
        
        # Perturbation strategy: if we had a "best" (least violations), we could use it.
        # But for now, random restarts.
        trial += 1
    
    return best_voltages
>>>>>>> REPLACE
