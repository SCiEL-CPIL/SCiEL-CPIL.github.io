<<<<<<< SEARCH
def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    def elapsed():
        return time.time() - t0
    
    # Phase 1: Try different initial structures
    best_adj_bits, best_deg, best_tris, best_score = _build_turan_plus_one()
    
    # Try various bipartition sizes
    for a_size in range(5, 12):
        if elapsed() > DEADLINE - 60:
            break
        result = _try_internal_edges(a_size, rng, n_trials=30)
        if result and result[3] < best_score:
            best_adj_bits, best_deg, best_tris, best_score = result
    
    # Also try some random non-bipartite starts
    for trial in range(10):
        if elapsed() > DEADLINE - 60:
            break
        # Start from random graph with 65 edges
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random_edges = set(rng.sample(all_pairs, TARGET_EDGES))
        adj_try = _build_adj_from_edges(random_edges)
        deg_try = _compute_degrees(adj_try)
        tris_try = _enumerate_triangles(adj_try)
        score_try = _max_tri_deg_sum(adj_try, deg_try, tris_try)
        if score_try < best_score:
            best_adj_bits, best_deg, best_tris, best_score = (
                adj_try, deg_try, tris_try, score_try)
    
    # Phase 2: Simulated Annealing with multiple restarts
    time_budget_sa = DEADLINE - elapsed() - 15.0  # reserve 15s for final polish
    if time_budget_sa > 5.0:
        # Determine number of SA restarts based on available time
        n_restarts = max(1, int(time_budget_sa / 20.0))
        time_per_restart = time_budget_sa / n_restarts
        
        for restart in range(n_restarts):
            if elapsed() > DEADLINE - 15.0:
                break
            
            # For restarts > 0, perturb the best solution
            if restart == 0:
                adj_bits = best_adj_bits[:]
                deg = best_deg[:]
                tris = best_tris[:]
                cur_score = best_score
            else:
                # Perturb: do k random swaps
                adj_bits = best_adj_bits[:]
                deg = _compute_degrees(adj_bits)
                tris = _enumerate_triangles(adj_bits)
                cur_score = _max_tri_deg_sum(adj_bits, deg, tris)
                
                all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
                edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
                nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
                
                for _ in range(5):
                    if not edges or not nonedges:
                        break
                    e_remove = edges[rng.randrange(len(edges))]
                    e_add = nonedges[rng.randrange(len(nonedges))]
                    new_adj, new_deg, new_score, new_tris = _apply_swap(
                        adj_bits, deg, e_remove, e_add)
                    adj_bits, deg, tris = new_adj, new_deg, new_tris
                    cur_score = new_score
                    edges.remove(e_remove)
                    nonedges.append(e_remove)
                    nonedges.remove(e_add)
                    edges.append(e_add)
            
            # SA parameters
            start_temp = 3.0
            end_temp = 0.01
            max_iters = 50000
            cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
            temp = start_temp
            
            all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
            nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
            
            iters = 0
            deadline_inner = DEADLINE - 15.0
            while iters < max_iters:
                if time.time() > deadline_inner:
                    break
                
                if not edges or not nonedges:
                    break
                
                e_remove = edges[rng.randrange(len(edges))]
                e_add = nonedges[rng.randrange(len(nonedges))]
                
                new_adj_bits, new_deg, new_score, new_tris = _apply_swap(
                    adj_bits, deg, e_remove, e_add)
                
                # Acceptance
                if new_score < cur_score:
                    accept = True
                elif new_score == cur_score:
                    accept = rng.random() < 0.3
                else:
                    delta = new_score - cur_score
                    if temp < 1e-10:
                        accept = False
                    else:
                        accept = math.exp(-delta / temp) > rng.random()
                
                if accept:
                    adj_bits = new_adj_bits
                    deg = new_deg
                    tris = new_tris
                    cur_score = new_score
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_adj_bits = adj_bits[:]
                        best_deg = deg[:]
                        best_tris = tris[:]
                    
                    # O(1) swap using index tracking
                    # Find positions
                    er_idx = edges.index(e_remove)
                    edges[er_idx] = e_add
                    nonedges.remove(e_add)
                    nonedges.append(e_remove)
                    # Fix: we need to move e_remove from edges to nonedges and e_add from nonedges to edges
                    # Actually simpler: just rebuild if needed, but for N=16 with 65 edges,
                    # list operations are cheap. Let's use a different approach.
                    
                iters += 1
                temp = max(end_temp, temp * cooling)
    
    # Phase 3: Deterministic hill-climbing polish (multiple passes)
    if elapsed() < DEADLINE - 3.0:
        adj_bits = best_adj_bits[:]
        deg = best_deg[:]
        tris = best_tris[:]
        cur_score = best_score
        
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
        nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
        
        # Multiple passes of best-improvement hill climbing
        for polish_pass in range(8):
            if elapsed() > DEADLINE - 3.0:
                break
            
            improved = False
            # Try all edge swaps, keep best improvement
            best_swap = None
            best_new_score = cur_score
            best_new_adj = None
            best_new_deg = None
            best_new_tris = None
            
            for e_remove in edges:
                if elapsed() > DEADLINE - 3.0:
                    break
                for e_add in nonedges:
                    new_adj_bits, new_deg, new_score, new_tris = _apply_swap(
                        adj_bits, deg, e_remove, e_add)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = (e_remove, e_add)
                        best_new_adj = new_adj_bits
                        best_new_deg = new_deg
                        best_new_tris = new_tris
            
            if best_swap is not None and best_new_score < cur_score:
                e_remove, e_add = best_swap
                adj_bits = best_new_adj
                deg = best_new_deg
                tris = best_new_tris
                cur_score = best_new_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_adj_bits = adj_bits[:]
                    best_deg = deg[:]
                    best_tris = tris[:]
                
                # O(1) swap using last-element trick
                er_pos = edges.index(e_remove)
                edges[er_pos] = edges[-1]
                edges.pop()
                ea_pos = nonedges.index(e_add)
                nonedges[ea_pos] = nonedges[-1]
                nonedges.pop()
                nonedges.append(e_remove)
                edges.append(e_add)
                improved = True
            
            if not improved:
                break
    
    # Final exhaustive 1-swap local search
    if elapsed() < DEADLINE - 2.0:
        adj_bits = best_adj_bits[:]
        deg = _compute_degrees(adj_bits)
        cur_score = _max_tri_deg_sum(adj_bits, deg, _enumerate_triangles(adj_bits))
        
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
        nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
        
        # Try every possible single swap
        for e_remove in edges:
            if elapsed() > DEADLINE - 2.0:
                break
            for e_add in nonedges:
                new_adj_bits, new_deg, new_score, _ = _apply_swap(
                    adj_bits, deg, e_remove, e_add)
                if new_score < cur_score:
                    # Apply this swap and restart the search
                    adj_bits = new_adj_bits
                    deg = new_deg
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_adj_bits = adj_bits[:]
                        best_deg = deg[:]
                    # Rebuild edge lists
                    edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
                    nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
                    break
    
    # Final verification
    final_deg = _compute_degrees(best_adj_bits)
    final_tris = _enumerate_triangles(best_adj_bits)
    final_score = _max_tri_deg_sum(best_adj_bits, final_deg, final_tris)
    
    result_edges = [[i, j] for i in range(N) for j in range(i + 1, N)
                    if (best_adj_bits[i] >> j) & 1]
    
    # Verify edge count
    assert len(result_edges) == TARGET_EDGES, f"Expected {TARGET_EDGES} edges, got {len(result_edges)}"
    
    return {'n': N, 'edges': result_edges}
=======
def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    def elapsed():
        return time.time() - t0
    
    # Phase 1: Try different initial structures
    best_adj_bits, best_deg, best_tris, best_score = _build_turan_plus_one()
    
    # Try various bipartition sizes
    for a_size in range(5, 12):
        if elapsed() > DEADLINE - 60:
            break
        result = _try_internal_edges(a_size, rng, n_trials=30)
        if result and result[3] < best_score:
            best_adj_bits, best_deg, best_tris, best_score = result
    
    # Also try some random non-bipartite starts
    for trial in range(10):
        if elapsed() > DEADLINE - 60:
            break
        # Start from random graph with 65 edges
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random_edges = set(rng.sample(all_pairs, TARGET_EDGES))
        adj_try = _build_adj_from_edges(random_edges)
        deg_try = _compute_degrees(adj_try)
        tris_try = _enumerate_triangles(adj_try)
        score_try = _max_tri_deg_sum(adj_try, deg_try, tris_try)
        if score_try < best_score:
            best_adj_bits, best_deg, best_tris, best_score = (
                adj_try, deg_try, tris_try, score_try)
    
    # Phase 2: Directed Local Search (DLS)
    # Strategy: Find the worst triangle, try to reduce the degree of its highest-degree vertex.
    # This is a structural move guided by the objective function.
    
    time_budget_dls = DEADLINE - elapsed() - 10.0
    if time_budget_dls > 5.0:
        
        # We will run DLS for the remaining time
        # Keep track of current state
        adj_bits = best_adj_bits[:]
        deg = best_deg[:]
        tris = best_tris[:]
        cur_score = best_score
        
        # Pre-compute edge and non-edge lists for fast access
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        def get_edges_nonedges(adj):
            edges = [(i, j) for (i, j) in all_pairs if (adj[i] >> j) & 1]
            nonedges = [(i, j) for (i, j) in all_pairs if not ((adj[i] >> j) & 1)]
            return edges, nonedges
        
        edges, nonedges = get_edges_nonedges(adj_bits)
        
        # DLS Loop
        while elapsed() < DEADLINE - 10.0:
            # 1. Identify the worst triangle
            if not tris:
                break
            
            worst_tri = max(tris, key=lambda t: deg[t[0]] + deg[t[1]] + deg[t[2]])
            worst_sum = deg[worst_tri[0]] + deg[worst_tri[1]] + deg[worst_tri[2]]
            
            # 2. Identify the vertex in the worst triangle with the highest degree
            # Tie-break by random order to avoid bias
            vertices_in_tri = list(worst_tri)
            rng.shuffle(vertices_in_tri)
            vertices_in_tri.sort(key=lambda v: deg[v], reverse=True)
            target_v = vertices_in_tri[0]
            
            # 3. Generate candidate moves:
            #    Remove an edge (target_v, u) and add (target_v, w) where w is a non-neighbor.
            #    This keeps deg(target_v) constant but changes degrees of u and w.
            #    Ideally, we want to reduce the degree of the other vertices in the triangle
            #    or move the "load" to vertices not in high-score triangles.
            
            # Get neighbors and non-neighbors of target_v
            neighbors = [j for j in range(N) if (adj_bits[target_v] >> j) & 1 and j != target_v]
            non_neighbors = [j for j in range(N) if not ((adj_bits[target_v] >> j) & 1) and j != target_v]
            
            if not neighbors or not non_neighbors:
                break
            
            # Shuffle to try random moves
            rng.shuffle(neighbors)
            rng.shuffle(non_neighbors)
            
            # Try up to 50 moves per iteration
            improved = False
            for _ in range(50):
                if not neighbors or not non_neighbors:
                    break
                
                u = neighbors[rng.randrange(len(neighbors))]
                w = non_neighbors[rng.randrange(len(non_neighbors))]
                
                e_remove = (target_v, u)
                e_add = (target_v, w)
                
                # Check if e_remove is actually an edge and e_add is a non-edge
                if not ((adj_bits[target_v] >> u) & 1):
                    continue
                if (adj_bits[target_v] >> w) & 1:
                    continue
                
                new_adj_bits, new_deg, new_score, new_tris = _apply_swap(
                    adj_bits, deg, e_remove, e_add)
                
                # Acceptance criterion:
                # 1. If new_score < cur_score, accept.
                # 2. If new_score == cur_score, accept with probability 0.5 (to explore).
                # 3. If new_score > cur_score, accept with small probability (annealing-like) to escape.
                
                if new_score < cur_score:
                    accept = True
                elif new_score == cur_score:
                    accept = rng.random() < 0.5
                else:
                    # Small chance to accept worse move to escape local optima
                    accept = rng.random() < 0.05
                
                if accept:
                    # Update state
                    adj_bits = new_adj_bits
                    deg = new_deg
                    tris = new_tris
                    cur_score = new_score
                    
                    # Update global best
                    if cur_score < best_score:
                        best_score = cur_score
                        best_adj_bits = adj_bits[:]
                        best_deg = deg[:]
                        best_tris = tris[:]
                    
                    # Update edge/non-edge lists
                    edges, nonedges = get_edges_nonedges(adj_bits)
                    improved = True
                    break # Restart the worst-triangle identification
            
            if not improved:
                # If no move accepted, do a random perturbation to escape
                # Pick a random edge and non-edge and swap
                if edges and nonedges:
                    e_remove = edges[rng.randrange(len(edges))]
                    e_add = nonedges[rng.randrange(len(nonedges))]
                    new_adj_bits, new_deg, new_score, new_tris = _apply_swap(
                        adj_bits, deg, e_remove, e_add)
                    adj_bits = new_adj_bits
                    deg = new_deg
                    tris = new_tris
                    cur_score = new_score
                    edges, nonedges = get_edges_nonedges(adj_bits)

    # Phase 3: Final Polish (Standard Hill Climbing)
    if elapsed() < DEADLINE - 2.0:
        adj_bits = best_adj_bits[:]
        deg = _compute_degrees(adj_bits)
        cur_score = _max_tri_deg_sum(adj_bits, deg, _enumerate_triangles(adj_bits))
        
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
        nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
        
        # Try every possible single swap to find a local optimum
        improved = True
        while improved and elapsed() < DEADLINE - 2.0:
            improved = False
            best_swap = None
            best_new_score = cur_score
            best_new_adj = None
            best_new_deg = None
            best_new_tris = None
            
            for e_remove in edges:
                if elapsed() > DEADLINE - 2.0:
                    break
                for e_add in nonedges:
                    new_adj_bits, new_deg, new_score, new_tris = _apply_swap(
                        adj_bits, deg, e_remove, e_add)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = (e_remove, e_add)
                        best_new_adj = new_adj_bits
                        best_new_deg = new_deg
                        best_new_tris = new_tris
            
            if best_swap is not None and best_new_score < cur_score:
                e_remove, e_add = best_swap
                adj_bits = best_new_adj
                deg = best_new_deg
                tris = best_new_tris
                cur_score = best_new_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_adj_bits = adj_bits[:]
                    best_deg = deg[:]
                    best_tris = tris[:]
                
                # Update lists
                edges, nonedges = get_edges_nonedges(adj_bits)
                improved = True
    
    # Final verification
    final_deg = _compute_degrees(best_adj_bits)
    final_tris = _enumerate_triangles(best_adj_bits)
    final_score = _max_tri_deg_sum(best_adj_bits, final_deg, final_tris)
    
    result_edges = [[i, j] for i in range(N) for j in range(i + 1, N)
                    if (best_adj_bits[i] >> j) & 1]
    
    # Verify edge count
    assert len(result_edges) == TARGET_EDGES, f"Expected {TARGET_EDGES} edges, got {len(result_edges)}"
    
    return {'n': N, 'edges': result_edges}
>>>>>>> REPLACE
