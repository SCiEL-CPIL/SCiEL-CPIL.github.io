<<<<<<< SEARCH
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
            while iters < max_iters:
                if elapsed() > DEADLINE - 15.0:
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
                    
                    edges.remove(e_remove)
                    nonedges.append(e_remove)
                    nonedges.remove(e_add)
                    edges.append(e_add)
                
                iters += 1
                temp = max(end_temp, temp * cooling)
=======
    # Phase 2: Simulated Annealing with multiple restarts and biased moves
    time_budget_sa = DEADLINE - elapsed() - 15.0  # reserve 15s for final polish
    if time_budget_sa > 5.0:
        n_restarts = max(1, int(time_budget_sa / 20.0))
        
        for restart in range(n_restarts):
            if elapsed() > DEADLINE - 15.0:
                break
            
            if restart == 0:
                adj_bits = best_adj_bits[:]
                deg = best_deg[:]
                tris = best_tris[:]
                cur_score = best_score
            else:
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
            
            start_temp = 3.0
            end_temp = 0.01
            max_iters = 50000
            cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
            temp = start_temp
            
            all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
            nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
            
            iters = 0
            while iters < max_iters:
                if elapsed() > DEADLINE - 15.0:
                    break
                
                if not edges or not nonedges:
                    break
                
                # Biased move: try to remove an edge incident to the worst triangle
                worst_tri = _get_worst_triangle(tris, deg)
                if worst_tri and rng.random() < 0.5:
                    # Pick a vertex from the worst triangle
                    v_worst = worst_tri[rng.randrange(3)]
                    # Find neighbors of v_worst in the current edge list
                    neighbors = [u for (u, v) in edges if u == v_worst or v == v_worst]
                    if neighbors:
                        # Pick a neighbor to remove the edge to
                        target_neigh = neighbors[rng.randrange(len(neighbors))]
                        e_remove = tuple(sorted((v_worst, target_neigh)))
                    else:
                        e_remove = edges[rng.randrange(len(edges))]
                else:
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
                    
                    edges.remove(e_remove)
                    nonedges.append(e_remove)
                    nonedges.remove(e_add)
                    edges.append(e_add)
                
                iters += 1
                temp = max(end_temp, temp * cooling)
>>>>>>> REPLACE
