<<<<<<< SEARCH
def coefficients(stages):
    """
    12-stage product formula using 2nd-order-matching initial guess and
    multi-start Nelder-Mead with Hermitian eigendecompositions.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Precompute Hermitian eigendecompositions (faster than general eig)
    A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs = _precompute_eigh(A, B, H)
    
    def eval_score(a, b):
        """Evaluate with bounds and sum checks."""
        if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
            return 1e10
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        return _objective_fast(a, b, A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs, dim)
    
    # Generate initial candidates
    candidates = []
    
    # 1. 2nd-order-matching symmetric coefficients (from _solve_2nd_order_constraints)
    a_2nd, b_2nd = _solve_2nd_order_constraints(stages)
    candidates.append((a_2nd.copy(), b_2nd.copy()))
    
    # 2. Equal weights baseline
    eq = np.ones(stages) / stages
    candidates.append((eq.copy(), eq.copy()))
    
    # 3. Symmetric pattern with different spread
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - 2.0 * x1
    
    # Alternative symmetric 12-step: [x3, x1, x1, x3, x1, x1, x1, x3, x1, x1, x3, x1]
    a_sym2 = np.array([x3, x1, x1, x3, x1, x1, x1, x3, x1, x1, x3, x1])
    b_sym2 = np.array([x2, x2, 0,  x2, x2, 0,  0,  x2, x2, 0,  x2, x2])
    a_sym2 = a_sym2 / np.sum(a_sym2)
    b_sym2 = b_sym2 / np.sum(b_sym2)
    candidates.append((a_sym2.copy(), b_sym2.copy()))
    
    # 4. Center-weighted
    center = np.arange(stages) - stages / 2.0
    a_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    b_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    a_cw = a_cw / np.sum(a_cw)
    b_cw = b_cw / np.sum(b_cw)
    candidates.append((a_cw.copy(), b_cw.copy()))
    
    # Evaluate all candidates
    best_a = candidates[0][0].copy()
    best_b = candidates[0][1].copy()
    best_score = eval_score(best_a, best_b)
    
    for a_c, b_c in candidates[1:]:
        s = eval_score(a_c, b_c)
        if s < best_score:
            best_a = a_c.copy()
            best_b = b_c.copy()
            best_score = s
    
    # Nelder-Mead with reduced parameterization (22 free params)
    def constrained_obj(params):
        a_free = params[:11]
        b_free = params[11:22]
        a_full = np.append(a_free, 1.0 - np.sum(a_free))
        b_full = np.append(b_free, 1.0 - np.sum(b_free))
        return eval_score(a_full, b_full)
    
    # Multi-start Nelder-Mead
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x0 = np.concatenate([best_a[:11], best_b[:11]])
        else:
            x0 = np.concatenate([best_a[:11], best_b[:11]]) + np.random.normal(0, 0.15, 22)
        
        # Initialize simplex
        n = 22
        simplex = [x0.copy()]
        for i in range(n):
            v = x0.copy()
            v[i] += 0.03 * (abs(v[i]) + 1.0)
            simplex.append(v)
        
        fvals = [constrained_obj(v) for v in simplex]
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = 300
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[idx] for idx in order]
            fvals = [fvals[idx] for idx in order]
            
            if abs(fvals[0] - fvals[-1]) < 1e-12:
                break
            
            a_full = np.append(simplex[0][:11], 1.0 - np.sum(simplex[0][:11]))
            b_full = np.append(simplex[0][11:22], 1.0 - np.sum(simplex[0][11:22]))
            if fvals[0] < best_score:
                best_a = a_full.copy()
                best_b = b_full.copy()
                best_score = fvals[0]
            
            centroid = np.mean(simplex[:-1], axis=0)
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = constrained_obj(x_r)
            
            if f_r < fvals[0]:
                x_e = centroid + gamma * (x_r - centroid)
                f_e = constrained_obj(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = constrained_obj(x_c)
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    for j in range(1, n + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        fvals[j] = constrained_obj(simplex[j])
    
    # Final cleanup
    best_a = np.clip(best_a, -50, 50)
    best_b = np.clip(best_b, -50, 50)
    best_a = best_a + (1.0 - np.sum(best_a)) / stages
    best_b = best_b + (1.0 - np.sum(best_b)) / stages
    
    if np.any(best_a < -50) or np.any(best_a > 50) or np.any(best_b < -50) or np.any(best_b > 50):
        best_a = np.ones(stages) / stages
        best_b = np.ones(stages) / stages
    
    return list(best_a), list(best_b)
=======
def coefficients(stages):
    """
    12-stage product formula using 2nd-order-matching initial guess and
    L-BFGS-B optimization with Hermitian eigendecompositions.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Precompute Hermitian eigendecompositions (faster than general eig)
    A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs = _precompute_eigh(A, B, H)
    
    def eval_score(a, b):
        """Evaluate with bounds and sum checks."""
        if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
            return 1e10
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        return _objective_fast(a, b, A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs, dim)
    
    # Generate initial candidates
    candidates = []
    
    # 1. 2nd-order-matching symmetric coefficients (from _solve_2nd_order_constraints)
    a_2nd, b_2nd = _solve_2nd_order_constraints(stages)
    candidates.append((a_2nd.copy(), b_2nd.copy()))
    
    # 2. Equal weights baseline
    eq = np.ones(stages) / stages
    candidates.append((eq.copy(), eq.copy()))
    
    # 3. Symmetric pattern with different spread
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - 2.0 * x1
    
    # Alternative symmetric 12-step: [x3, x1, x1, x3, x1, x1, x1, x3, x1, x1, x3, x1]
    a_sym2 = np.array([x3, x1, x1, x3, x1, x1, x1, x3, x1, x1, x3, x1])
    b_sym2 = np.array([x2, x2, 0,  x2, x2, 0,  0,  x2, x2, 0,  x2, x2])
    a_sym2 = a_sym2 / np.sum(a_sym2)
    b_sym2 = b_sym2 / np.sum(b_sym2)
    candidates.append((a_sym2.copy(), b_sym2.copy()))
    
    # 4. Center-weighted
    center = np.arange(stages) - stages / 2.0
    a_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    b_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    a_cw = a_cw / np.sum(a_cw)
    b_cw = b_cw / np.sum(b_cw)
    candidates.append((a_cw.copy(), b_cw.copy()))
    
    # Evaluate all candidates
    best_a = candidates[0][0].copy()
    best_b = candidates[0][1].copy()
    best_score = eval_score(best_a, best_b)
    
    for a_c, b_c in candidates[1:]:
        s = eval_score(a_c, b_c)
        if s < best_score:
            best_a = a_c.copy()
            best_b = b_c.copy()
            best_score = s
    
    # L-BFGS-B with reduced parameterization (22 free params)
    def constrained_obj(params):
        a_free = params[:11]
        b_free = params[11:22]
        a_full = np.append(a_free, 1.0 - np.sum(a_free))
        b_full = np.append(b_free, 1.0 - np.sum(b_free))
        return eval_score(a_full, b_full)
    
    # Multi-start L-BFGS-B
    np.random.seed(42)
    for restart in range(8):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            x0 = np.concatenate([best_a[:11], best_b[:11]])
        else:
            x0 = np.concatenate([best_a[:11], best_b[:11]]) + np.random.normal(0, 0.1, 22)
        
        # Bounds for each parameter
        bounds = [(-50, 50)] * 22
        
        try:
            result = minimize(
                constrained_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < best_score:
                a_full = np.append(result.x[:11], 1.0 - np.sum(result.x[:11]))
                b_full = np.append(result.x[11:22], 1.0 - np.sum(result.x[11:22]))
                if eval_score(a_full, b_full) < best_score:
                    best_a = a_full.copy()
                    best_b = b_full.copy()
                    best_score = result.fun
        except Exception:
            pass
    
    # Final cleanup
    best_a = np.clip(best_a, -50, 50)
    best_b = np.clip(best_b, -50, 50)
    best_a = best_a + (1.0 - np.sum(best_a)) / stages
    best_b = best_b + (1.0 - np.sum(best_b)) / stages
    
    if np.any(best_a < -50) or np.any(best_a > 50) or np.any(best_b < -50) or np.any(best_b > 50):
        best_a = np.ones(stages) / stages
        best_b = np.ones(stages) / stages
    
    return list(best_a), list(best_b)
>>>>>>> REPLACE
