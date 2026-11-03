<<<<<<< SEARCH
    # If we have time, do a few random restarts with SLSQP
    if time_left() > 20:
        for _ in range(10):
            if time_left() < 10:
                break
            try:
                pts = np.zeros((N, 2))
                for i in range(N):
                    u = rng.random()
                    v = rng.random()
                    while u + v > 1.0:
                        u = rng.random()
                        v = rng.random()
                    pts[i, 0] = u
                    pts[i, 1] = v
                init_bary = _project_to_bary_simplex(pts)
                
                x_slsqp = _local_optimize_slsqp(init_bary.flatten(), bounds, maxiter=100)
                bary_slsqp = _project_to_bary_simplex(x_slsqp.reshape(N, 2))
                score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                
                if score_slsqp > best_score:
                    best_score = score_slsqp
                    best_x = bary_slsqp
            except Exception:
                pass

    if best_x is None:
        best_x = _structured_init_fps()
        
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
=======
    # If we have time, do perturbation restarts from best solution
    if time_left() > 15 and best_x is not None:
        for _ in range(8):
            if time_left() < 10:
                break
            try:
                # Perturb the best solution
                perturbation_scale = 0.05
                perturbed = best_x + rng.normal(0, perturbation_scale, size=(N, 2))
                perturbed = _project_to_bary_simplex(perturbed)
                
                # Quick L-BFGS-B refinement
                current_x = perturbed.flatten()
                for beta in [50.0, 150.0]:
                    if time_left() < 8:
                        break
                    res = minimize(
                        _objective_smooth,
                        current_x,
                        args=(beta,),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
                    )
                    x_opt = _project_to_bary_simplex(res.x.reshape(N, 2))
                    score = _min_area_ratio_from_bary(x_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
                        current_x = x_opt.flatten()
            except Exception:
                pass

    # Final high-precision polish pass
    if best_x is not None and time_left() > 5:
        polish_budget = min(time_left() - 3, 15.0)
        if polish_budget > 2:
            try:
                polished = _polish_pass(best_x, bounds, time_budget=polish_budget)
                polished_score = _min_area_ratio_from_bary(polished)
                if polished_score > best_score:
                    best_score = polished_score
                    best_x = polished
            except Exception:
                pass

    if best_x is None:
        best_x = _structured_init_fps()
        
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    # Final safety check: ensure all points are inside the triangle
    u_final = cart_points[:, 0]
    v_final = cart_points[:, 1]
    w_final = 1.0 - u_final - v_final
    
    # Clip any tiny numerical errors
    eps = 1e-10
    u_final = np.clip(u_final, 0, 1)
    v_final = np.clip(v_final, 0, 1)
    w_final = np.clip(w_final, 0, 1)
    s = u_final + v_final + w_final
    u_final /= s
    v_final /= s
    w_final /= s
    
    cart_points = _bary_to_cart(u_final, v_final)
    
    return cart_points
>>>>>>> REPLACE
