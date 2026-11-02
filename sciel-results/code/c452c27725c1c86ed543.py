<<<<<<< SEARCH
    # Run multiple SA trials with different seeds
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        
        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        # Give each trial a fair time budget
        trial_deadline = min(deadline - 5, start_time + 140.0)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows
    
    # Final polish with remaining time
    if time.time() < deadline - 2:
        polish_deadline = min(deadline, start_time + 143.0)
        
        # First, do SA polish
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows
        
        # Then, do row re-optimization polish
        if time.time() < deadline - 1:
            reopt_rows, reopt_scope = row_reopt_polish(
                [r[:] for r in best_solution], n_rows, k, deadline - 1
            )
            if reopt_scope < best_scope:
                best_scope = reopt_scope
                best_solution = reopt_rows

    return best_solution
=======
    # Run multiple SA trials with different seeds
    num_trials = 15
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        # Give each trial a fair time budget
        trial_deadline = min(deadline - 5, start_time + 140.0)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows
    
    # Final polish with remaining time
    if time.time() < deadline - 2:
        polish_deadline = min(deadline, start_time + 143.0)
        
        # First, do SA polish
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows
        
        # Then, do row re-optimization polish
        if time.time() < deadline - 1:
            reopt_rows, reopt_scope = row_reopt_polish(
                [r[:] for r in best_solution], n_rows, k, deadline - 1
            )
            if reopt_scope < best_scope:
                best_scope = reopt_scope
                best_solution = reopt_rows
        
        # Final aggressive polish: multiple passes of row re-optimization
        if time.time() < deadline - 0.5:
            for _ in range(3):
                if time.time() > deadline - 0.5:
                    break
                reopt_rows, reopt_scope = row_reopt_polish(
                    [r[:] for r in best_solution], n_rows, k, deadline - 0.5
                )
                if reopt_scope < best_scope:
                    best_scope = reopt_scope
                    best_solution = reopt_rows
                else:
                    break

    return best_solution
>>>>>>> REPLACE
