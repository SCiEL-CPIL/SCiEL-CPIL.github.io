<<<<<<< SEARCH
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy baselines
    best_rows = None
    best_scope = float('inf')
    for seed_off in range(15):
        rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed_off)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # Phase 2: Randomized backtracking with decreasing bounds
    if time.time() < deadline - 40.0:
        bt_rows, bt_scope = _randomized_backtracking(n_rows, k, deadline - 40.0, rng, initial_max=best_scope)
        if bt_rows is not None and bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 15.0:
        sa_rows = _simulated_annealing(best_rows, k, deadline - 15.0, rng)
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 4: Local search to improve
    if time.time() < deadline - 5.0:
        improved = _local_search_improve(best_rows, k, deadline - 5.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 5: Final refinement with difference-swap mutations
    if time.time() < deadline - 2.0:
        used = _all_diffs(best_rows)
        for _ in range(15):
            if time.time() > deadline - 2.0:
                break
            refined, used = _difference_swap_mutation(best_rows, k, used, num_swaps=rng.randint(2, 5), rng=rng)
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
    
    return best_rows
=======
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy baselines
    best_rows = None
    best_scope = float('inf')
    for seed_off in range(15):
        rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed_off)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # Phase 2: Genetic algorithm (structural change)
    if time.time() < deadline - 30.0:
        ga_rows = _genetic_algorithm(best_rows, k, deadline - 30.0, rng, pop_size=15)
        if ga_rows is not None and _valid(ga_rows):
            ga_scope = _scope(ga_rows)
            if ga_scope < best_scope:
                best_rows = ga_rows
                best_scope = ga_scope
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 10.0:
        improved = _local_search_improve(best_rows, k, deadline - 10.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 4: Final refinement with difference-swap mutations
    if time.time() < deadline - 3.0:
        used = _all_diffs(best_rows)
        for _ in range(20):
            if time.time() > deadline - 3.0:
                break
            refined, used = _difference_swap_mutation(best_rows, k, used, num_swaps=rng.randint(2, 5), rng=rng)
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
    
    # Phase 5: Final local search
    if time.time() < deadline - 1.0:
        improved = _local_search_improve(best_rows, k, deadline - 1.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    return best_rows
>>>>>>> REPLACE
