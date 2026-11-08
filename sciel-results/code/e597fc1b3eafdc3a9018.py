<<<<<<< SEARCH
    # Try iterative column removal
    for iteration in range(100):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
        
        result = _try_remove_columns(best[0], best[1], best[2])
        if result is not None:
            best = result
            best_rank = best[0].shape[1]
            if best_rank <= 50:  # Good enough, keep going
                continue
        else:
            break
=======
    # Try iterative column removal
    for iteration in range(200):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        result = _try_remove_columns(best[0], best[1], best[2])
        if result is not None:
            best = result
            best_rank = best[0].shape[1]
            continue
        
        # Try combination absorption
        result2 = _try_column_combination_absorption(best[0], best[1], best[2], TIME_LIMIT - 10)
        if result2 is not None:
            best = result2
            best_rank = best[0].shape[1]
            continue
        
        # Try random perturbations to enable further removal
        result3 = _random_perturb_and_reduce(best[0], best[1], best[2], n_iter=5)
        if result3[0].shape[1] < best_rank:
            best = result3
            best_rank = best[0].shape[1]
            continue
        
        if elapsed > TIME_LIMIT - 20:
            break
>>>>>>> REPLACE
