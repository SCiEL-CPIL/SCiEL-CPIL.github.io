<<<<<<< SEARCH
    # Phase 2: Simulated annealing with structured moves
    if best_adj is None:
        best_adj = build_adj([], [(0, 1)], [])
        best_score = get_score(best_adj)
    
    current_adj = best_adj.copy()
    current_score = best_score
    
    T = 3.0
    cooling = 0.9995
    min_T = 0.3
    
    # Precompute edge lists for faster access
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    edges.append((i, j))
        return edges
    
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))
        return edges
    
    # Structured moves: swap a cross edge for an internal edge or vice versa
    while time.time() - start_time < deadline - 20:
        # Choose move type
        r = np.random.random()
        
        if r < 0.4:
            # Remove a cross edge, add an internal edge
            cross_edges = [(u, v) for u, v in all_cross if current_adj[u, v] == 1]
            if not cross_edges:
                continue
            u, v = cross_edges[np.random.randint(len(cross_edges))]
            
            int_options = [(a, b) for a, b in all_int_A if current_adj[a, b] == 0] + \
                         [(a, b) for a, b in all_int_B if current_adj[a, b] == 0]
            if not int_options:
                continue
            x, y = int_options[np.random.randint(len(int_options))]
            
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[x, y] = 1
            current_adj[y, x] = 1
            
            new_score = get_score(current_adj)
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0
                    
        elif r < 0.7:
            # Remove an internal edge, add a cross edge
            int_edges = [(a, b) for a, b in all_int_A if current_adj[a, b] == 1] + \
                        [(a, b) for a, b in all_int_B if current_adj[a, b] == 1]
            if not int_edges:
                continue
            x, y = int_edges[np.random.randint(len(int_edges))]
            
            cross_options = [(u, v) for u, v in all_cross if current_adj[u, v] == 0]
            if not cross_options:
                continue
            u, v = cross_options[np.random.randint(len(cross_options))]
            
            current_adj[x, y] = 0
            current_adj[y, x] = 0
            current_adj[u, v] = 1
            current_adj[v, u] = 1
            
            new_score = get_score(current_adj)
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    current_adj[x, y] = 1
                    current_adj[y, x] = 1
                    current_adj[u, v] = 0
                    current_adj[v, u] = 0
                    
        else:
            # General edge swap
            edges = get_edges(current_adj)
            if not edges:
                continue
            u, v = edges[np.random.randint(len(edges))]
            
            non_edges = get_non_edges(current_adj)
            if not non_edges:
                continue
            x, y = non_edges[np.random.randint(len(non_edges))]
            
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[x, y] = 1
            current_adj[y, x] = 1
            
            new_score = get_score(current_adj)
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0
        
        T *= cooling
        if T < min_T:
            T = min_T
=======
    # Phase 2: Simulated annealing with structured moves (optimized)
    if best_adj is None:
        best_adj = build_adj([], [(0, 1)], [])
        best_score = get_score(best_adj)
    
    # Precompute arrays for fast access
    cross_arr = np.array(all_cross)
    intA_arr = np.array(all_int_A)
    intB_arr = np.array(all_int_B)
    
    # Multi-restart SA
    num_restarts = 5
    restart_budget = (deadline - 25) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > deadline - 25:
            break
        
        if restart == 0:
            current_adj = best_adj.copy()
            current_score = best_score
        else:
            # Start from best, perturb slightly
            current_adj = best_adj.copy()
            # Randomly swap 1-3 edge pairs
            for _ in range(np.random.randint(1, 4)):
                r = np.random.random()
                if r < 0.5:
                    # Remove cross, add internal
                    ce_idx = np.where(current_adj[cross_arr[:, 0], cross_arr[:, 1]] == 1)[0]
                    if len(ce_idx) == 0:
                        continue
                    ci = ce_idx[np.random.randint(len(ce_idx))]
                    u, v = cross_arr[ci]
                    
                    io_idx = np.where(current_adj[intA_arr[:, 0], intA_arr[:, 1]] == 0)[0]
                    io_idx2 = np.where(current_adj[intB_arr[:, 0], intB_arr[:, 1]] == 0)[0]
                    all_io = list(zip(io_idx, io_idx2))
                    if not all_io:
                        # Try just A
                        if len(io_idx) > 0:
                            ii = io_idx[np.random.randint(len(io_idx))]
                            x, y = intA_arr[ii]
                            current_adj[u, v] = 0
                            current_adj[v, u] = 0
                            current_adj[x, y] = 1
                            current_adj[y, x] = 1
                        continue
                    # Pick from combined
                    if len(io_idx) > 0 and (len(io_idx2) == 0 or np.random.random() < 0.5):
                        ii = io_idx[np.random.randint(len(io_idx))]
                        x, y = intA_arr[ii]
                    else:
                        ii = io_idx2[np.random.randint(len(io_idx2))]
                        x, y = intB_arr[ii]
                    current_adj[u, v] = 0
                    current_adj[v, u] = 0
                    current_adj[x, y] = 1
                    current_adj[y, x] = 1
                else:
                    # Remove internal, add cross
                    ie_idx = np.where(current_adj[intA_arr[:, 0], intA_arr[:, 1]] == 1)[0]
                    ie_idx2 = np.where(current_adj[intB_arr[:, 0], intB_arr[:, 1]] == 1)[0]
                    if len(ie_idx) == 0 and len(ie_idx2) == 0:
                        continue
                    if len(ie_idx) > 0 and (len(ie_idx2) == 0 or np.random.random() < 0.5):
                        ii = ie_idx[np.random.randint(len(ie_idx))]
                        x, y = intA_arr[ii]
                    else:
                        ii = ie_idx2[np.random.randint(len(ie_idx2))]
                        x, y = intB_arr[ii]
                    
                    co_idx = np.where(current_adj[cross_arr[:, 0], cross_arr[:, 1]] == 0)[0]
                    if len(co_idx) == 0:
                        continue
                    ci = co_idx[np.random.randint(len(co_idx))]
                    u, v = cross_arr[ci]
                    
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
            
            current_score = get_score(current_adj)
        
        T = 2.5
        cooling = 0.9998
        min_T = 0.2
        restart_end = time.time() + restart_budget
        
        while time.time() < restart_end:
            r = np.random.random()
            
            if r < 0.45:
                # Remove a cross edge, add an internal edge
                ce_mask = current_adj[cross_arr[:, 0], cross_arr[:, 1]] == 1
                ce_idx = np.where(ce_mask)[0]
                if len(ce_idx) == 0:
                    continue
                ci = ce_idx[np.random.randint(len(ce_idx))]
                u, v = cross_arr[ci]
                
                ia_mask = current_adj[intA_arr[:, 0], intA_arr[:, 1]] == 0
                ib_mask = current_adj[intB_arr[:, 0], intB_arr[:, 1]] == 0
                ia_idx = np.where(ia_mask)[0]
                ib_idx = np.where(ib_mask)[0]
                
                if len(ia_idx) == 0 and len(ib_idx) == 0:
                    continue
                if len(ia_idx) > 0 and (len(ib_idx) == 0 or np.random.random() < 0.5):
                    ii = ia_idx[np.random.randint(len(ia_idx))]
                    x, y = intA_arr[ii]
                else:
                    ii = ib_idx[np.random.randint(len(ib_idx))]
                    x, y = intB_arr[ii]
                
                current_adj[u, v] = 0
                current_adj[v, u] = 0
                current_adj[x, y] = 1
                current_adj[y, x] = 1
                
                new_score = fast_score(current_adj)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                else:
                    delta = new_score - current_score
                    if np.random.random() < np.exp(-delta / T):
                        current_score = new_score
                    else:
                        current_adj[u, v] = 1
                        current_adj[v, u] = 1
                        current_adj[x, y] = 0
                        current_adj[y, x] = 0
                        
            elif r < 0.75:
                # Remove an internal edge, add a cross edge
                ia_mask = current_adj[intA_arr[:, 0], intA_arr[:, 1]] == 1
                ib_mask = current_adj[intB_arr[:, 0], intB_arr[:, 1]] == 1
                ia_idx = np.where(ia_mask)[0]
                ib_idx = np.where(ib_mask)[0]
                
                if len(ia_idx) == 0 and len(ib_idx) == 0:
                    continue
                if len(ia_idx) > 0 and (len(ib_idx) == 0 or np.random.random() < 0.5):
                    ii = ia_idx[np.random.randint(len(ia_idx))]
                    x, y = intA_arr[ii]
                else:
                    ii = ib_idx[np.random.randint(len(ib_idx))]
                    x, y = intB_arr[ii]
                
                co_mask = current_adj[cross_arr[:, 0], cross_arr[:, 1]] == 0
                co_idx = np.where(co_mask)[0]
                if len(co_idx) == 0:
                    continue
                ci = co_idx[np.random.randint(len(co_idx))]
                u, v = cross_arr[ci]
                
                current_adj[x, y] = 0
                current_adj[y, x] = 0
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                
                new_score = fast_score(current_adj)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                else:
                    delta = new_score - current_score
                    if np.random.random() < np.exp(-delta / T):
                        current_score = new_score
                    else:
                        current_adj[x, y] = 1
                        current_adj[y, x] = 1
                        current_adj[u, v] = 0
                        current_adj[v, u] = 0
                        
            else:
                # General edge swap - pick any edge and any non-edge
                all_pairs = np.array([(i, j) for i in range(n) for j in range(i+1, n)])
                edge_mask = current_adj[all_pairs[:, 0], all_pairs[:, 1]] == 1
                nonedge_mask = current_adj[all_pairs[:, 0], all_pairs[:, 1]] == 0
                
                e_idx = np.where(edge_mask)[0]
                ne_idx = np.where(nonedge_mask)[0]
                
                if len(e_idx) == 0 or len(ne_idx) == 0:
                    continue
                ei = e_idx[np.random.randint(len(e_idx))]
                ni = ne_idx[np.random.randint(len(ne_idx))]
                
                u, v = all_pairs[ei]
                x, y = all_pairs[ni]
                
                current_adj[u, v] = 0
                current_adj[v, u] = 0
                current_adj[x, y] = 1
                current_adj[y, x] = 1
                
                new_score = fast_score(current_adj)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                else:
                    delta = new_score - current_score
                    if np.random.random() < np.exp(-delta / T):
                        current_score = new_score
                    else:
                        current_adj[u, v] = 1
                        current_adj[v, u] = 1
                        current_adj[x, y] = 0
                        current_adj[y, x] = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
>>>>>>> REPLACE

