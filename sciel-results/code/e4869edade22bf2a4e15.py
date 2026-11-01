<<<<<<< SEARCH
    # Phase 2: High-precision polish on best solution
    if best_result is not None and time.time() < deadline - 10:
        polished = refine_solution(best_result, max_iter=1000, tol=1e-15, penalty_weight=5000)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Try perturbations and re-refine
        for polish_iter in range(15):
            if time.time() > deadline - 5:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            xy_p += rng.uniform(-0.003, 0.003, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=50)
            polished2 = refine_solution(perturbed, max_iter=500, tol=1e-15, penalty_weight=5000)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()
=======
    # Phase 2: High-precision polish on best solution
    if best_result is not None and time.time() < deadline - 10:
        # Newton refinement for high precision
        polished = newton_refine(best_result, max_iter=300, penalty_weight=10000, tol=1e-15)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Try perturbations and re-refine with Newton
        for polish_iter in range(20):
            if time.time() > deadline - 5:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            xy_p += rng.uniform(-0.002, 0.002, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=50)
            polished2 = newton_refine(perturbed, max_iter=200, penalty_weight=10000, tol=1e-15)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

        # Final radius growth with Newton
        if time.time() < deadline - 5:
            grown = grow_radii(best_result, n_grow=100, growth_rate=0.001)
            if is_valid(grown):
                score = get_score(grown)
                if score > best_score:
                    best_score = score
                    best_result = grown.copy()
>>>>>>> REPLACE
