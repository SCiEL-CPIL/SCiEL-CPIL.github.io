<<<<<<< SEARCH
    # Final Polish - two stages for better convergence
    if time.time() < deadline - 5.0:
        # Stage 1: Medium precision
        final_refined = _refine_l_bfgs_b(best_centers, beta=500.0, maxiter=300)
        final_score = _raw_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_refined
    
    if time.time() < deadline - 2.0:
        # Stage 2: High precision
        final_refined = _refine_l_bfgs_b(best_centers, beta=800.0, maxiter=400)
        final_score = _raw_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_refined
=======
    # Final Polish - three stages for better convergence
    if time.time() < deadline - 5.0:
        # Stage 1: Medium precision
        final_refined = _refine_l_bfgs_b(best_centers, beta=500.0, maxiter=300)
        final_score = _raw_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_refined
    
    if time.time() < deadline - 3.0:
        # Stage 2: High precision
        final_refined = _refine_l_bfgs_b(best_centers, beta=800.0, maxiter=400)
        final_score = _raw_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_refined
    
    if time.time() < deadline - 1.0:
        # Stage 3: Newton polish for exact contact satisfaction
        newton_final = _newton_polish(best_centers, max_iter=50)
        newton_score = _raw_score(newton_final)
        if newton_score > best_score:
            best_score = newton_score
            best_centers = newton_final
>>>>>>> REPLACE
