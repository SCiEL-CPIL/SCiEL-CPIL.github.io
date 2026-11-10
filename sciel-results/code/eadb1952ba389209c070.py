<<<<<<< SEARCH
    out = []
    start_time = time.time()
    total_budget = 140.0 # Leave 10s margin for 150s limit
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        inst_start = time.time()
        # Check if we have enough time for this instance
        # Assume roughly equal time per instance, but adjust based on remaining
        remaining_time = total_budget - (time.time() - start_time)
        if remaining_time <= 0:
            # Fallback: return best possible (empty or all vertices) immediately
            # But we need to generate the graph at least? No, we can just return [] if we know it's invalid?
            # No, we must return a valid transversal.
            # If we run out of time, we should return a known valid transversal.
            # The set of all vertices is always a valid transversal.
            # So we can just return list(range(n)) as a fallback.
            out.append(list(range(n)))
            continue
            
        # Allocate time for this instance
        # If it's the first instance, give it more time?
        # Let's just use a fixed fraction of remaining time.
        # But we need to be careful.
        # Let's say we have T remaining, and K instances left.
        # Give T/K to each?
        # But some instances might be harder.
        # Let's just give 35s per instance as before, but capped by remaining time.
        time_limit = min(35.0, remaining_time)
        
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            out.append([])
            continue
            
        best_set = None
        best_size = n # Upper bound
        
        # Initial greedy solution
        rng_init = random.Random(seed)
        greedy_sol = greedy_set_cover(cliques, n, rng_init)
        greedy_sol = local_search(greedy_sol, cliques, rng_init, max_steps=100)
        
        if len(greedy_sol) < best_size:
            best_size = len(greedy_sol)
            best_set = set(greedy_sol)
            
        # Iterative improvement
        rng = random.Random(12345)
        iteration = 0
        while time.time() - inst_start < time_limit:
            # New greedy start with different seed
            rng_iter = random.Random(seed + iteration)
            candidate = greedy_set_cover(cliques, n, rng_iter)
            
            # Local search
            candidate = local_search(candidate, cliques, rng_iter, max_steps=200)
            
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = set(candidate)
                
                # If we found a very small set, we can stop early?
                # Or continue to see if we can do better.
                # Let's continue.
                
            # Simulated annealing on the current best
            if best_set is not None:
                sa_sol = simulated_annealing(best_set, cliques, rng_iter, start_temp=1.0, end_temp=0.1, steps=500)
                if len(sa_sol) < best_size:
                    best_size = len(sa_sol)
                    best_set = sa_sol
                    
            iteration += 1
            
            # Check time again
            if time.time() - inst_start > time_limit:
                break
                
        if best_set is None:
            best_set = set(range(n))
            
        out.append(sorted(best_set))
        
    return out
=======
    out = []
    start_time = time.time()
    total_budget = 145.0 # Leave 5s margin for 150s limit
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        inst_start = time.time()
        remaining_time = total_budget - (time.time() - start_time)
        if remaining_time <= 0:
            out.append(list(range(n)))
            continue
            
        # Allocate time: more for larger graphs
        if idx == 0:
            time_limit = min(40.0, remaining_time)
        elif idx == 1:
            time_limit = min(45.0, remaining_time)
        else:
            time_limit = min(50.0, remaining_time)
        
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            out.append([])
            continue
            
        best_set = None
        best_size = n + 1
        
        # Try branch and bound first (with time limit)
        bb_sol = branch_and_bound_set_cover(cliques, n, time_limit=10.0)
        if bb_sol:
            best_size = len(bb_sol)
            best_set = set(bb_sol)
        
        # If B&B didn't find anything or found something large, use greedy
        if best_set is None or best_size > n // 2:
            rng_init = random.Random(seed)
            greedy_sol = greedy_set_cover(cliques, n, rng_init)
            greedy_sol = local_search(greedy_sol, cliques, rng_init, max_steps=200)
            if len(greedy_sol) < best_size:
                best_size = len(greedy_sol)
                best_set = set(greedy_sol)
        
        # Iterative improvement with local search and simulated annealing
        rng = random.Random(12345)
        iteration = 0
        while time.time() - inst_start < time_limit:
            # New greedy start with different seed
            rng_iter = random.Random(seed + iteration * 7)
            candidate = greedy_set_cover(cliques, n, rng_iter)
            
            # Local search
            candidate = local_search(candidate, cliques, rng_iter, max_steps=300)
            
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = set(candidate)
            
            # Simulated annealing on the current best
            if best_set is not None:
                sa_sol = simulated_annealing(best_set, cliques, rng_iter, start_temp=1.0, end_temp=0.1, steps=800)
                if len(sa_sol) < best_size:
                    best_size = len(sa_sol)
                    best_set = sa_sol
            
            iteration += 1
            
            # Check time again
            if time.time() - inst_start > time_limit:
                break
                
        if best_set is None:
            best_set = set(range(n))
            
        out.append(sorted(best_set))
        
    return out
>>>>>>> REPLACE
