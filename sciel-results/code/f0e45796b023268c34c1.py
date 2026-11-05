<<<<<<< SEARCH
    # Optimize each candidate
    for mats, ths in candidates:
        if time.time() > deadline - 10:
            break
        
        x_opt = _optimize_stack(mats, ths, problem, wls, n_mat, deadline - 5, max_layer, max_total)
        
        # Update best
        update_best(mats, x_opt)
        
        # Also try with the original thicknesses in case optimization failed
        update_best(mats, ths)
    
    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
=======
    # Optimize each candidate
    for mats, ths in candidates:
        if time.time() > deadline - 30:
            break
        
        x_opt = _optimize_stack(mats, ths, problem, wls, n_mat, deadline - 25, max_layer, max_total)
        
        # Update best
        update_best(mats, x_opt)
        
        # Also try with the original thicknesses in case optimization failed
        update_best(mats, ths)
    
    # Run genetic algorithm on the best candidate found so far
    if best_mats is not None and time.time() < deadline - 15:
        try:
            ga_mats, ga_ths = _genetic_algorithm(
                best_mats, best_ths, problem, wls, n_mat,
                deadline - 10, max_layer, max_total,
                pop_size=25, n_generations=40
            )
            update_best(ga_mats, ga_ths)
        except Exception:
            pass
    
    # Final high-precision polish on the best candidate
    if best_mats is not None and time.time() < deadline - 3:
        try:
            polished_ths = _final_polish(
                best_mats, best_ths, problem, wls, n_mat,
                deadline - 1, max_layer, max_total
            )
            update_best(best_mats, polished_ths)
        except Exception:
            pass
    
    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
>>>>>>> REPLACE
