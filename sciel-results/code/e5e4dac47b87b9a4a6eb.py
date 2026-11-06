<<<<<<< SEARCH
    no_improve_count = 0
    max_no_improve = 800
    
    while time.time() < deadline - 8 and no_improve_count < max_no_improve:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        if has_k4_fast(adj):
            adj = old_adj
            no_improve_count += 1
            continue
            
        m = compute_triangle_packing_bound(adj)
        ub = n - m
        
        if ub >= current_score + 2:
            adj = old_adj
            no_improve_count += 1
            continue
            
        new_score = max_triangle_free_size(adj, time_limit=0.7)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_adj_sa = adj[:]
            no_improve_count = 0
        else:
            adj = old_adj
            no_improve_count += 1
            
        if no_improve_count > 300 and random.random() < 0.15:
            for _ in range(random.randint(2, 4)):
                i2 = random.randint(0, n - 1)
                j2 = random.randint(0, n - 1)
                if i2 == j2:
                    continue
                if i2 > j2:
                    i2, j2 = j2, i2
                if (adj[i2] >> j2) & 1:
                    adj[i2] &= ~(1 << j2)
                    adj[j2] &= ~(1 << i2)
                else:
                    adj[i2] |= (1 << j2)
                    adj[j2] |= (1 << i2)
            if has_k4_fast(adj):
                adj = best_adj_sa[:]
                current_score = best_score_sa
            else:
                m = compute_triangle_packing_bound(adj)
                ub = n - m
                if ub >= current_score + 2:
                    adj = best_adj_sa[:]
                    current_score = best_score_sa
                    no_improve_count += 1
                    continue
                    
                new_score = max_triangle_free_size(adj, time_limit=0.7)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score_sa:
                        best_score_sa = current_score
                        best_adj_sa = adj[:]
                    no_improve_count = 0
                else:
                    adj = best_adj_sa[:]
                    current_score = best_score_sa
                    no_improve_count += 1
=======
    no_improve_count = 0
    max_no_improve = 800
    
    # Track which clique each vertex belongs to for structured moves
    clique_of = [0]*6 + [1]*6 + [2]*6
    
    while time.time() < deadline - 8 and no_improve_count < max_no_improve:
        # Choose move type: 70% random edge, 30% structured inter-clique move
        if random.random() < 0.7:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
        else:
            # Structured: pick two vertices from different cliques
            ci = random.randint(0, 2)
            cj = random.choice([0, 1, 2])
            while cj == ci:
                cj = random.randint(0, 2)
            i = random.randint(ci*6, ci*6+5)
            j = random.randint(cj*6, cj*6+5)
        
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        if has_k4_fast(adj):
            adj = old_adj
            no_improve_count += 1
            continue
            
        m = compute_triangle_packing_bound(adj)
        ub = n - m
        
        if ub >= current_score + 2:
            adj = old_adj
            no_improve_count += 1
            continue
            
        new_score = max_triangle_free_size(adj, time_limit=0.7)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_adj_sa = adj[:]
            no_improve_count = 0
        else:
            adj = old_adj
            no_improve_count += 1
            
        if no_improve_count > 300 and random.random() < 0.15:
            # Multi-edge perturbation with bias toward inter-clique edges
            for _ in range(random.randint(2, 4)):
                if random.random() < 0.6:
                    ci = random.randint(0, 2)
                    cj = random.choice([0, 1, 2])
                    while cj == ci:
                        cj = random.randint(0, 2)
                    i2 = random.randint(ci*6, ci*6+5)
                    j2 = random.randint(cj*6, cj*6+5)
                else:
                    i2 = random.randint(0, n - 1)
                    j2 = random.randint(0, n - 1)
                if i2 == j2:
                    continue
                if i2 > j2:
                    i2, j2 = j2, i2
                if (adj[i2] >> j2) & 1:
                    adj[i2] &= ~(1 << j2)
                    adj[j2] &= ~(1 << i2)
                else:
                    adj[i2] |= (1 << j2)
                    adj[j2] |= (1 << i2)
            if has_k4_fast(adj):
                adj = best_adj_sa[:]
                current_score = best_score_sa
            else:
                m = compute_triangle_packing_bound(adj)
                ub = n - m
                if ub >= current_score + 2:
                    adj = best_adj_sa[:]
                    current_score = best_score_sa
                    no_improve_count += 1
                    continue
                    
                new_score = max_triangle_free_size(adj, time_limit=0.7)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score_sa:
                        best_score_sa = current_score
                        best_adj_sa = adj[:]
                    no_improve_count = 0
                else:
                    adj = best_adj_sa[:]
                    current_score = best_score_sa
                    no_improve_count += 1
>>>>>>> REPLACE
