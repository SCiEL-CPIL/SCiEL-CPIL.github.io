<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate candidate pool - focus on smooth numbers for high collision density
    pool = set(range(1, 51))
    pool.update(generate_smooth_numbers([2, 3, 5], 10000))
    pool.update(generate_smooth_numbers([2, 3, 5, 7], 10000))
    for p in [2, 3, 5, 7, 11]:
        v = 1
        while v <= 10000:
            pool.add(v)
            v *= p
    
    pool = sorted([x for x in pool if 1 <= x <= 10000])
    
    # Build collision graph
    W, vals = build_collision_graph(pool)
    
    # Get initial set via greedy modularity
    initial_set = greedy_modularity_selection(W, vals, k=20)
    
    # Also try branch-and-bound clique search
    random.seed(42)
    other_initials = [
        list(range(1, 21)),
        generate_smooth_numbers([2, 3, 5], 10000)[:20],
        generate_smooth_numbers([2, 3, 5, 7], 10000)[:20],
        sorted(random.sample(pool, 20))
    ]
    
    # Add branch-and-bound result
    bnb_deadline = start_time + 30.0
    bnb_result = branch_and_bound_clique(W, 20, bnb_deadline, max_nodes=40)
    if len(bnb_result) == 20:
        other_initials.append(sorted([vals[i] for i in bnb_result]))
    
    best_score = float('inf')
    best_A = initial_set
    
    # Evaluate all initial sets
    candidates = [initial_set] + other_initials
    for s in candidates:
        s = sorted(set(s))
        if len(s) == 20:
            score = compute_metrics(s)
            if score < best_score:
                best_score = score
                best_A = s[:]
    
    # Polish with local search
    if time.time() < deadline - 5:
        best_A = local_search_polish(best_A, pool, deadline - 3)
    
    # Final check
    final_score = compute_metrics(best_A)
    if final_score < best_score:
        best_score = final_score
    
    return best_A
=======
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Generate candidate pool - focus on smooth numbers for high collision density
    pool = set(range(1, 51))
    pool.update(generate_smooth_numbers([2, 3, 5], 10000))
    pool.update(generate_smooth_numbers([2, 3, 5, 7], 10000))
    for p in [2, 3, 5, 7, 11]:
        v = 1
        while v <= 10000:
            pool.add(v)
            v *= p
    
    pool = sorted([x for x in pool if 1 <= x <= 10000])
    
    # Build collision graph
    W, vals = build_collision_graph(pool)
    
    # Get initial set via greedy modularity
    initial_set = greedy_modularity_selection(W, vals, k=20)
    
    # Also try branch-and-bound clique search
    random.seed(42)
    other_initials = [
        list(range(1, 21)),
        generate_smooth_numbers([2, 3, 5], 10000)[:20],
        generate_smooth_numbers([2, 3, 5, 7], 10000)[:20],
        sorted(random.sample(pool, 20))
    ]
    
    # Add branch-and-bound result
    bnb_deadline = start_time + 20.0
    bnb_result = branch_and_bound_clique(W, 20, bnb_deadline, max_nodes=40)
    if len(bnb_result) == 20:
        other_initials.append(sorted([vals[i] for i in bnb_result]))
    
    best_score = float('inf')
    best_A = initial_set
    
    # Evaluate all initial sets
    candidates = [initial_set] + other_initials
    for s in candidates:
        s = sorted(set(s))
        if len(s) == 20:
            score = compute_metrics(s)
            if score < best_score:
                best_score = score
                best_A = s[:]
    
    # Use multi-objective evolutionary algorithm
    if time.time() < deadline - 5:
        # Prepare diverse initial population
        initial_population = [best_A]
        for s in candidates:
            s_sorted = sorted(set(s))
            if len(s_sorted) == 20:
                initial_population.append(s_sorted)
        
        # Add some random diverse individuals
        for _ in range(10):
            initial_population.append(sorted(random.sample(pool, 20)))
        
        evolved_A = multi_objective_evolution(initial_population, pool, deadline - 3)
        
        if evolved_A is not None:
            evolved_score = compute_metrics(evolved_A)
            if evolved_score < best_score:
                best_score = evolved_score
                best_A = evolved_A
    
    # Final local search polish on the best solution found
    if time.time() < deadline - 2:
        polished_A = local_search_polish(best_A, pool, deadline - 1)
        polished_score = compute_metrics(polished_A)
        if polished_score < best_score:
            best_score = polished_score
            best_A = polished_A
    
    # Final check
    final_score = compute_metrics(best_A)
    if final_score < best_score:
        best_score = final_score
    
    return best_A
>>>>>>> REPLACE
