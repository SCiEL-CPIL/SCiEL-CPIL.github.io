    def simulated_annealing(adj, degs, start_temp=10.0, end_temp=0.1, time_limit=10.0):
        start = time.time()
        score = calc_score(adj, degs)
        best_score = score
        best_adj = [row[:] for row in adj]
        best_degs = degs[:]
        
        while time.time() - start < time_limit:
            # Pick random edge to remove
            edges = get_edges(adj)
            non_edges = get_non_edges(adj)
            if not edges or not non_edges: break
            
            rem_u, rem_v = random.choice(edges)
            add_u, add_v = random.choice(non_edges)
            
            if (rem_u, rem_v) == (add_u, add_v) or (rem_u, rem_v) == (add_v, add_u):
                continue
                
            # Apply
            adj[rem_u][rem_v] = adj[rem_v][rem_u] = False
            adj[add_u][add_v] = adj[add_v][add_u] = True
            
            new_degs = degs[:]
            new_degs[rem_u] -= 1
            new_degs[rem_v] -= 1
            new_degs[add_u] += 1
            new_degs[add_v] += 1
            
            new_score = calc_score(adj, new_degs)
            
            # Acceptance
            if new_score < score or random.random() < math.exp(-(new_score - score) / max(start_temp, 1e-9)):
                score = new_score
                degs = new_degs
                if score < best_score:
                    best_score = score
                    best_adj = [row[:] for row in adj]
                    best_degs = degs[:]
            else:
                # Revert
                adj[rem_u][rem_v] = adj[rem_v][rem_u] = True
                adj[add_u][add_v] = adj[add_v][add_u] = False
                
            # Cool down
            start_temp *= 0.999
        return best_adj, best_score
    