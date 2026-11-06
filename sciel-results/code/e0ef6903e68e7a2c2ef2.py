<<<<<<< SEARCH
    # 3. Main loop: multiple restarts with different initializations
    rng_global = random.Random(42)
    best_global_s = None
    best_global_score = float('inf')

    # Budget allocation:
    # - First run: Thue-Morse init (strong heuristic)
    # - Second run: Random init
    # - Remaining runs: Random inits, each with decreasing time budget

    # Run 1: Thue-Morse
    tm = thue_morse(N)
    t1 = time.time()
    s1, score1 = greedy_optimize(tm, time_limit * 0.4, rng_global)
    if score1 < best_global_score:
        best_global_score = score1
        best_global_s = s1.copy()

    # Run 2: Random init
    if time.time() - start_time < time_limit * 0.7:
        rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s2, score2 = greedy_optimize(rand_init, time_limit * 0.3, rng_global)
        if score2 < best_global_score:
            best_global_score = score2
            best_global_s = s2.copy()

    # Remaining runs: more random inits with shorter budgets
    remaining_budget = time_limit - (time.time() - start_time)
    if remaining_budget > 2.0:
        # Do a few more short runs
        num_extra = min(5, int(remaining_budget / 3.0))
        for k in range(num_extra):
            if time.time() - start_time > time_limit - 1.0:
                break
            rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            s_k, score_k = greedy_optimize(rand_init, min(3.0, time_limit - (time.time() - start_time) - 1.0), rng_global)
            if score_k < best_global_score:
                best_global_score = score_k
                best_global_s = s_k.copy()

    # Final safety: if we haven't found anything (shouldn't happen), return random
    if best_global_s is None:
        best_global_s = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_global_s]
=======
    # 3. Main loop: multiple restarts with different initializations
    rng_global = random.Random(42)
    best_global_s = None
    best_global_score = float('inf')

    def evaluate_signs(s):
        """Compute max absolute AP sum for a signs array."""
        sums = compute_sums(s)
        return int(np.max(np.abs(sums)))

    # Budget allocation:
    # - Run 1: Thue-Morse init + greedy (strong heuristic)
    # - Run 2: Thue-Morse init + continuous optimization
    # - Run 3: Random init + continuous optimization
    # - Remaining: greedy on best found so far with perturbations

    # Run 1: Thue-Morse + greedy
    tm = thue_morse(N)
    s1, score1 = greedy_optimize(tm, time_limit * 0.35, rng_global)
    if score1 < best_global_score:
        best_global_score = score1
        best_global_s = s1.copy()

    # Run 2: Thue-Morse + continuous optimization
    if time.time() - start_time < time_limit * 0.65:
        s2 = continuous_optimize(tm, time_limit * 0.30, rng_global)
        score2 = evaluate_signs(s2)
        if score2 < best_global_score:
            best_global_score = score2
            best_global_s = s2.copy()

    # Run 3: Random init + continuous optimization
    if time.time() - start_time < time_limit * 0.80:
        rand_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s3 = continuous_optimize(rand_init, time_limit * 0.15, rng_global)
        score3 = evaluate_signs(s3)
        if score3 < best_global_score:
            best_global_score = score3
            best_global_s = s3.copy()

    # Run 4+: Greedy refinement on best found, with random perturbations
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0 and best_global_s is not None:
        # Do a few short greedy runs perturbing the best solution
        num_extra = min(4, int(remaining / 2.5))
        for k in range(num_extra):
            if time.time() - start_time > time_limit - 1.0:
                break
            # Perturb best solution
            perturbed = best_global_s.copy()
            num_flips = rng_global.randint(5, 20)
            flip_positions = rng_global.sample(range(N), num_flips)
            for pos in flip_positions:
                perturbed[pos] = -perturbed[pos]
            budget_k = min(2.5, time_limit - (time.time() - start_time) - 1.0)
            if budget_k < 0.5:
                break
            s_k, score_k = greedy_optimize(perturbed, budget_k, rng_global)
            if score_k < best_global_score:
                best_global_score = score_k
                best_global_s = s_k.copy()

    # Final safety
    if best_global_s is None:
        best_global_s = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_global_s]
>>>>>>> REPLACE
