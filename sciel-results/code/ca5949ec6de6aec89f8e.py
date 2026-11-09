<<<<<<< SEARCH
    best_a, best_b = _get_4th_order_init()
    best_err = _compute_error(best_a, best_b, A, B, H_eig)
    
    # Convert to symmetric params
    p0 = (best_a[0] + best_a[11]) / 2
    p1 = (best_a[1] + best_a[10]) / 2
    p2 = (best_a[2] + best_a[9]) / 2
    p3 = (best_a[3] + best_a[8]) / 2
    p4 = (best_a[4] + best_a[7]) / 2
    p5 = (best_a[5] + best_a[6]) / 2
    q0 = (best_b[0] + best_b[11]) / 2
    q1 = (best_b[1] + best_b[10]) / 2
    q2 = (best_b[2] + best_b[9]) / 2
    q3 = (best_b[3] + best_b[8]) / 2
    q4 = (best_b[4] + best_b[7]) / 2
    q5 = (best_b[5] + best_b[6]) / 2
    
    x0 = np.array([p0, p1, p2, p3, p4, p5, q0, q1, q2, q3, q4, q5])
    
    # Phase 1: Fast optimization with Frobenius norm proxy
    if _remaining() > 30:
        def fast_objective(params):
            a, b = _symmetric_params_to_ab(params)
            a, b = _normalize_ab(a, b)
            return _compute_frobenius_error(a, b, A, B, H_eig)
        
        try:
            result = minimize(
                fast_objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-50, 50)] * 12,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success and _remaining() > 15:
                a_opt, b_opt = _symmetric_params_to_ab(result.x)
                a_opt, b_opt = _normalize_ab(a_opt, b_opt)
                err_opt = _compute_error(a_opt, b_opt, A, B, H_eig)
                if err_opt < best_err:
                    best_err = err_opt
                    best_a = a_opt
                    best_b = b_opt
        except Exception:
            pass
    
    # Phase 2: Fine-tune with spectral norm using Nelder-Mead
    if _remaining() > 15:
        def objective(params):
            a, b = _symmetric_params_to_ab(params)
            a, b = _normalize_ab(a, b)
            return _compute_error(a, b, A, B, H_eig)
        
        # Re-initialize simplex around current best
        p0 = (best_a[0] + best_a[11]) / 2
        p1 = (best_a[1] + best_a[10]) / 2
        p2 = (best_a[2] + best_a[9]) / 2
        p3 = (best_a[3] + best_a[8]) / 2
        p4 = (best_a[4] + best_a[7]) / 2
        p5 = (best_a[5] + best_a[6]) / 2
        q0 = (best_b[0] + best_b[11]) / 2
        q1 = (best_b[1] + best_b[10]) / 2
        q2 = (best_b[2] + best_b[9]) / 2
        q3 = (best_b[3] + best_b[8]) / 2
        q4 = (best_b[4] + best_b[7]) / 2
        q5 = (best_b[5] + best_b[6]) / 2
        
        x0 = np.array([p0, p1, p2, p3, p4, p5, q0, q1, q2, q3, q4, q5])
        
        n = 12
        simplex = [x0.tolist()]
        for i in range(n):
            xi = x0.tolist()
            xi[i] += 0.0005
            simplex.append(xi)
        
        fvals = [objective(x) for x in simplex]
        
        max_iter = 200
        for it in range(max_iter):
            if _remaining() < 5:
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[n])
            f_ref = objective(x_ref)
            
            if f_ref < fvals[0]:
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[n])
                f_exp = objective(x_exp)
                if f_exp < f_ref:
                    simplex[n] = x_exp.tolist()
                    fvals[n] = f_exp
                else:
                    simplex[n] = x_ref.tolist()
                    fvals[n] = f_ref
            elif f_ref < fvals[n-1]:
                simplex[n] = x_ref.tolist()
                fvals[n] = f_ref
            else:
                rho = 0.5
                x_con = centroid + rho * (simplex[n] - centroid)
                f_con = objective(x_con)
                if f_con < fvals[n]:
                    simplex[n] = x_con.tolist()
                    fvals[n] = f_con
                else:
                    for i in range(1, n+1):
                        simplex[i] = (simplex[0] + 0.5 * (np.array(simplex[i]) - simplex[0])).tolist()
                        fvals[i] = objective(simplex[i])
        
        best_idx = np.argmin(fvals)
        best_params = simplex[best_idx]
        a_opt, b_opt = _symmetric_params_to_ab(best_params)
        a_opt, b_opt = _normalize_ab(a_opt, b_opt)
        
        err_opt = _compute_error(a_opt, b_opt, A, B, H_eig)
        if err_opt < best_err:
            best_err = err_opt
            best_a = a_opt
            best_b = b_opt
    
    # Final normalization
    best_a, best_b = _normalize_ab(best_a, best_b)
    
    return best_a, best_b
=======
    best_a, best_b = _get_4th_order_init()
    best_err = _compute_error(best_a, best_b, A, B, H_eig)
    
    # Use 24 independent parameters (no symmetry)
    x0 = np.array(best_a + best_b)
    n_params = 24
    
    def objective(params):
        a, b = _params_to_ab(params)
        a, b = _normalize_ab(a, b)
        return _compute_error(a, b, A, B, H_eig)
    
    # Coordinate Descent with line search
    current_params = x0.copy()
    current_err = best_err
    
    # Initial step size
    step_size = 0.05
    
    while _remaining() > 10:
        improved = False
        for i in range(n_params):
            if _remaining() < 5:
                break
            
            # Line search along dimension i
            # Try a few steps in both directions
            best_local_err = current_err
            best_local_param = current_params[i]
            
            for delta in [step_size, step_size/2, step_size/4, -step_size, -step_size/2, -step_size/4]:
                trial_params = current_params.copy()
                trial_params[i] += delta
                trial_a, trial_b = _params_to_ab(trial_params)
                trial_a, trial_b = _normalize_ab(trial_a, trial_b)
                trial_err = _compute_error(trial_a, trial_b, A, B, H_eig)
                
                if trial_err < best_local_err:
                    best_local_err = trial_err
                    best_local_param = trial_params[i]
            
            if best_local_param != current_params[i]:
                current_params[i] = best_local_param
                current_err = best_local_err
                improved = True
                if current_err < best_err:
                    best_err = current_err
                    best_a, best_b = _params_to_ab(current_params)
                    best_a, best_b = _normalize_ab(best_a, best_b)
        
        # If no improvement, reduce step size
        if not improved:
            step_size *= 0.5
            if step_size < 1e-5:
                break
        else:
            # Slightly increase step size if we improved a lot? Keep simple.
            pass
    
    # Final normalization
    best_a, best_b = _normalize_ab(best_a, best_b)
    
    return best_a, best_b
>>>>>>> REPLACE
