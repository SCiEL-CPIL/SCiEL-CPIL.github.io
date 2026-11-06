<<<<<<< SEARCH
    # Spectral-guided local search phase
    # Use max eigenvalue of adjacency matrix as a smooth surrogate
    # In quasi-regular graphs, spectral radius correlates with max degree sum
    if time.time() < deadline - 3:
        def spectral_radius_from_adj(adj_masks):
            """Compute max eigenvalue of adjacency matrix via power iteration."""
            A = np.zeros((n, n), dtype=np.float64)
            for u in range(n):
                mu = adj_masks[u]
                for v in range(u + 1, n):
                    if mu & (1 << v):
                        A[u, v] = 1.0
                        A[v, u] = 1.0
            # Power iteration
            x = np.ones(n, dtype=np.float64)
            x /= np.linalg.norm(x)
            for _ in range(200):
                x_new = A @ x
                norm = np.linalg.norm(x_new)
                if norm < 1e-15:
                    break
                x_new /= norm
                if np.abs(np.dot(x_new, x) - 1.0) < 1e-12:
                    x = x_new
                    break
                x = x_new
            return float(np.dot(x, A @ x))
        
        # Spectral-guided hill climbing
        current_es = edges_to_set(best_edges)
        current_adj, current_deg = build_adj(None, current_es)
        current_score = best_score
        current_spec = spectral_radius_from_adj(current_adj)
        
        random.seed(777)
        no_improve_spec = 0
        max_no_improve_spec = 3000
        
        while time.time() < deadline - 3 and no_improve_spec < max_no_improve_spec:
            e_remove = random.choice(list(current_es))
            candidates = [p for p in all_pairs if p not in current_es]
            if not candidates:
                break
            e_add = random.choice(candidates)
            
            new_es = set(current_es)
            new_es.remove(e_remove)
            new_es.add(e_add)
            
            new_adj, new_deg = build_adj(None, new_es)
            new_spec = spectral_radius_from_adj(new_adj)
            new_score = score_from_adj(new_adj, new_deg)
            
            # Accept if spectral radius improves OR exact score improves
            if new_spec < current_spec or new_score < current_score:
                current_es = new_es
                current_adj, current_deg = new_adj, new_deg
                current_score = new_score
                current_spec = new_spec
                no_improve_spec = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges_to_list(current_es)
            else:
                no_improve_spec += 1
        
        # If spectral search found something better, restart local search from there
        if best_score >= 24 and time.time() < deadline - 3:
            # Quick local search from the spectrally-improved solution
            local_es = edges_to_set(best_edges)
            local_adj, local_deg = build_adj(None, local_es)
            local_score = best_score
            random.seed(4242)
            for _ in range(5000):
                if time.time() > deadline - 3:
                    break
                e_remove = random.choice(list(local_es))
                candidates = [p for p in all_pairs if p not in local_es]
                if not candidates:
                    break
                e_add = random.choice(candidates)
                new_es = set(local_es)
                new_es.remove(e_remove)
                new_es.add(e_add)
                new_adj, new_deg = build_adj(None, new_es)
                new_score = score_from_adj(new_adj, new_deg)
                if new_score < local_score:
                    local_es = new_es
                    local_adj, local_deg = new_adj, new_deg
                    local_score = new_score
                    if local_score < best_score:
                        best_score = local_score
                        best_edges = edges_to_list(local_es)
=======
    # Degree-regularity guided local search phase
    # Use variance of degrees as a cheap smooth surrogate (lower variance = more regular)
    # Regular graphs tend to have lower max triangle degree sums
    if time.time() < deadline - 3:
        def degree_variance(adj_masks, deg):
            """Compute variance of degrees as a cheap regularity measure."""
            mean_deg = sum(deg) / n
            return sum((d - mean_deg) ** 2 for d in deg) / n
        
        current_es = edges_to_set(best_edges)
        current_adj, current_deg = build_adj(None, current_es)
        current_score = best_score
        current_var = degree_variance(current_adj, current_deg)
        
        random.seed(777)
        no_improve_reg = 0
        max_no_improve_reg = 5000
        
        while time.time() < deadline - 3 and no_improve_reg < max_no_improve_reg:
            e_remove = random.choice(list(current_es))
            candidates = [p for p in all_pairs if p not in current_es]
            if not candidates:
                break
            e_add = random.choice(candidates)
            
            new_es = set(current_es)
            new_es.remove(e_remove)
            new_es.add(e_add)
            
            new_adj, new_deg = build_adj(None, new_es)
            new_var = degree_variance(new_adj, new_deg)
            new_score = score_from_adj(new_adj, new_deg)
            
            # Accept if degree variance improves OR exact score improves
            if new_var < current_var or new_score < current_score:
                current_es = new_es
                current_adj, current_deg = new_adj, new_deg
                current_score = new_score
                current_var = new_var
                no_improve_reg = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges_to_list(current_es)
            else:
                no_improve_reg += 1
        
        # If regularity search found something better, restart local search from there
        if best_score >= 24 and time.time() < deadline - 3:
            # Quick local search from the improved solution
            local_es = edges_to_set(best_edges)
            local_adj, local_deg = build_adj(None, local_es)
            local_score = best_score
            random.seed(4242)
            for _ in range(8000):
                if time.time() > deadline - 3:
                    break
                e_remove = random.choice(list(local_es))
                candidates = [p for p in all_pairs if p not in local_es]
                if not candidates:
                    break
                e_add = random.choice(candidates)
                new_es = set(local_es)
                new_es.remove(e_remove)
                new_es.add(e_add)
                new_adj, new_deg = build_adj(None, new_es)
                new_score = score_from_adj(new_adj, new_deg)
                if new_score < local_score:
                    local_es = new_es
                    local_adj, local_deg = new_adj, new_deg
                    local_score = new_score
                    if local_score < best_score:
                        best_score = local_score
                        best_edges = edges_to_list(local_es)
>>>>>>> REPLACE
