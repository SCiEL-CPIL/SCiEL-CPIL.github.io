<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Diverse initializations
    initializations = [_initial_grid()]
    for s in range(6):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + best-fit local search from each init
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 30:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=30000, seed=init_idx * 7 + 1,
            deadline=deadline - 25
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 20:
            break

        # Best-fit local search (targeted moves)
        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=5000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 15
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        # Additional standard local search pass
        if time.time() > deadline - 12:
            break
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=2000, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 10
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Best-fit local search on best solution
    if time.time() < deadline - 15:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=6000,
            seed=777, deadline=deadline - 12
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 3: Standard local search with finer steps
    if time.time() < deadline - 10:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=4000, step_size=0.002,
            seed=999, deadline=deadline - 8
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Gradient-based polish (smooth surrogate) - multiple passes
    for pass_i in range(3):
        if time.time() < deadline - 8:
            polished = _gradient_polish(best_pos, deadline - 6)
            if len(polished) == 3:
                p_pol, r_pol, obj_pol = polished
                if obj_pol > best_obj:
                    best_pos = p_pol
                    best_obj = obj_pol
                else:
                    break

    # Phase 5: High-precision final polish (progressive beta)
    if time.time() < deadline - 5:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 2)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 6: Another gradient polish after high-precision
    if time.time() < deadline - 3:
        polished2 = _gradient_polish(best_pos, deadline - 1)
        if len(polished2) == 3:
            p_pol2, r_pol2, obj_pol2 = polished2
            if obj_pol2 > best_obj:
                best_pos = p_pol2
                best_obj = obj_pol2

    # Phase 7: Final fine-tuning local search
    if time.time() < deadline - 1:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=1000, step_size=0.001,
            seed=12345, deadline=deadline - 0.5
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Diverse initializations: grid, hex, and random
    initializations = [_initial_grid(), _initial_hex()]
    for s in range(8):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + local search from each init
    n_inits = len(initializations)
    sa_time_budget = (deadline - 35 - start_time) / n_inits
    sa_budget = max(sa_time_budget, 2.0)

    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 35:
            break

        sa_deadline = min(time.time() + sa_budget, deadline - 30)
        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=50000, seed=init_idx * 7 + 1,
            deadline=sa_deadline
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 25:
            break

        # Best-fit local search (targeted moves)
        bf_deadline = min(time.time() + 5.0, deadline - 18)
        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=8000,
            seed=init_idx * 13 + 2,
            deadline=bf_deadline
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        # Additional standard local search pass
        if time.time() > deadline - 15:
            break
        ls_deadline = min(time.time() + 3.0, deadline - 12)
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=3000, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=ls_deadline
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Best-fit local search on best solution
    if time.time() < deadline - 15:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=10000,
            seed=777, deadline=deadline - 12
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 3: Standard local search with finer steps
    if time.time() < deadline - 10:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=5000, step_size=0.002,
            seed=999, deadline=deadline - 8
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Gradient-based polish (smooth surrogate) - multiple passes
    for pass_i in range(4):
        if time.time() < deadline - 8:
            polished = _gradient_polish(best_pos, deadline - 6)
            if len(polished) == 3:
                p_pol, r_pol, obj_pol = polished
                if obj_pol > best_obj:
                    best_pos = p_pol
                    best_obj = obj_pol
                else:
                    break

    # Phase 5: High-precision final polish (progressive beta)
    if time.time() < deadline - 5:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 2)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 6: Another gradient polish after high-precision
    if time.time() < deadline - 3:
        polished2 = _gradient_polish(best_pos, deadline - 1)
        if len(polished2) == 3:
            p_pol2, r_pol2, obj_pol2 = polished2
            if obj_pol2 > best_obj:
                best_pos = p_pol2
                best_obj = obj_pol2

    # Phase 7: Final fine-tuning local search
    if time.time() < deadline - 1:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=1500, step_size=0.001,
            seed=12345, deadline=deadline - 0.5
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result
>>>>>>> REPLACE

Now let me also increase the L-BFGS-B maxiter in the gradient polish functions:

