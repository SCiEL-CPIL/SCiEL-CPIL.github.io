    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        
        # Find maximal cliques
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        
        valid_cliques = [c for c in cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        M = len(valid_cliques)
        
        # Map each vertex to a bitmask of cliques it belongs to
        masks = [0] * n
        for i, clique in enumerate(valid_cliques):
            bit = 1 << i
            for v in clique:
                masks[v] |= bit
                
        all_cliques_mask = (1 << M) - 1
        
        best_sol = None
        best_size = float('inf')
        
        # Greedy Set Cover to get initial solutions
        # We run a few restarts to get diverse initial solutions for Local Search
        initial_sols = []
        for _ in range(20):
            uncovered = all_cliques_mask
            selected = []
            
            while uncovered:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if not (uncovered & masks[v]):
                        continue
                    count = (uncovered & masks[v]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_v = v
                    elif count == best_count and count > 0:
                        if random.random() < 0.5:
                            best_v = v
                            
                if best_v == -1:
                    break
                    
                selected.append(best_v)
                uncovered &= ~masks[best_v]
                
            # Post-process: Remove redundant vertices
            changed = True
            while changed:
                changed = False
                for i in range(len(selected) - 1, -1, -1):
                    v = selected[i]
                    other_mask = 0
                    for j in range(len(selected)):
                        if j != i:
                            other_mask |= masks[selected[j]]
                    if not (masks[v] & ~other_mask):
                        selected.pop(i)
                        changed = True
                        break
            
            initial_sols.append(selected)
            
        # Run Local Search on the best initial solution
        # We pick the smallest initial solution to start
        initial_sols.sort(key=len)
        best_initial = initial_sols[0] if initial_sols else []
        
        # Local Search
        sol = set(best_initial)
        
        # Precompute cov_counts
        cov_counts = [0] * M
        for v in sol:
            m = masks[v]
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cov_counts[i] += 1
                m ^= lsb
                
        # Check validity (should be valid)
        # Run Local Search loop
        for _ in range(5000):
            improved = False
            
            # 1. Try to remove
            for v in list(sol):
                m = masks[v]
                critical = False
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    if cov_counts[i] == 1:
                        critical = True
                        break
                    m ^= lsb
                    
                if not critical:
                    sol.remove(v)
                    m = masks[v]
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        cov_counts[i] -= 1
                        m ^= lsb
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
            
            # 2. Try swaps
            if not improved:
                found_swap = False
                for v in list(sol):
                    U = []
                    m = masks[v]
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        if cov_counts[i] == 1:
                            U.append(i)
                        m ^= lsb
                        
                    if not U:
                        continue
                    
                    inter = set(valid_cliques[U[0]])
                    for idx in U[1:]:
                        inter &= set(valid_cliques[idx])
                        
                    candidates = [u for u in inter if u not in sol]
                    if candidates:
                        u = random.choice(candidates)
                        sol.remove(v)
                        sol.add(u)
                        
                        m = masks[v]
                        while m:
                            lsb = m & -m
                            i = lsb.bit_length() - 1
                            cov_counts[i] -= 1
                            m ^= lsb
                        m = masks[u]
                        while m:
                            lsb = m & -m
                            i = lsb.bit_length() - 1
                            cov_counts[i] += 1
                            m ^= lsb
                            
                        improved = True
                        found_swap = True
                        break
                if not found_swap:
                    break
                    
            if not improved:
                break
                
        if best_sol is None:
            best_sol = sorted(best_initial)
        else:
            best_sol = sorted(best_sol)
            
        return best_sol
