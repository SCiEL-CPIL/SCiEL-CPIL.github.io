import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Define Validation Cosmologies
    z_vals = np.unique(np.concatenate([
        np.logspace(np.log10(0.01), np.log10(1.0), 12),
        np.linspace(1.0, 10.0, 20)
    ]))
    
    w0_vals = np.linspace(-1.5, 0.5, 10)
    w1_vals = np.linspace(-1.5, 1.5, 10)
    
    validation_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                validation_cases.append((w0, w1, z))
                
    n_cases = len(validation_cases)
    
    # 2. Precompute Exact Integrals using fixed high-order GL quadrature (vectorized)
    # Use 128-node GL for reference — accurate to ~1e-15 for smooth integrands
    gl_ref_nodes, gl_ref_weights = np.polynomial.legendre.leggauss(128)
    u_ref = 0.5 + 0.5 * gl_ref_nodes  # map to [0,1]
    w_ref = 0.5 * gl_ref_weights
    
    w0_arr = np.array([c[0] for c in validation_cases])
    w1_arr = np.array([c[1] for c in validation_cases])
    z_arr = np.array([c[2] for c in validation_cases])
    
    # Vectorized integrand evaluation: shape (n_cases, 128)
    t_ref = z_arr[:, None] * u_ref[None, :]  # (n_cases, 128)
    
    log_t1 = 3.0 * np.log1p(t_ref)
    log_t2 = (3.0 * (1.0 + w0_arr))[:, None] * np.log1p(t_ref) + \
             (3.0 * w1_arr)[:, None] * (1.0 - 1.0 / (1.0 + t_ref))
    
    max_log = np.maximum(log_t1, log_t2)
    sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
    log_sum = max_log + np.log(sum_exp)
    f_ref = np.exp(-0.5 * log_sum)  # (n_cases, 128)
    
    # Exact integral = sum of f_ref * w_ref over the 128 nodes
    exact_ints = np.sum(f_ref * w_ref[None, :], axis=1)  # (n_cases,)
    
    # Check time
    if time.time() > deadline - 20:
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        u0 = 0.5 + 0.5 * gl_nodes
        w0_init = 0.5 * gl_weights
        return {"nodes": u0, "weights": w0_init}

    # 3. Define Objective Functions
    
    def compute_errors(params):
        """
        params: 16-dim vector.
        First 8: theta (nodes via sigmoid)
        Last 8: alpha (weights via softmax)
        """
        theta = params[:8]
        alpha = params[8:16]
        
        theta_clipped = np.clip(theta, -50, 50)
        u = 1.0 / (1.0 + np.exp(-theta_clipped))
        
        alpha_clipped = np.clip(alpha, -50, 50)
        exp_alpha = np.exp(alpha_clipped - np.max(alpha_clipped))
        w = exp_alpha / np.sum(exp_alpha)
        
        # Vectorized: shape (n_cases, 8)
        t = z_arr[:, None] * u[None, :]
        
        log_t1 = 3.0 * np.log1p(t)
        log_t2 = (3.0 * (1.0 + w0_arr))[:, None] * np.log1p(t) + \
                 (3.0 * w1_arr)[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_t1, log_t2)
        sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
        log_sum = max_log + np.log(sum_exp)
        
        f_vals = np.exp(-0.5 * log_sum)  # (n_cases, 8)
        
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        rel_errors = np.abs(quad_sums - exact_ints_safe) / exact_ints_safe
        
        return rel_errors, u, w

    def objective_max(params):
        """Hard max relative error"""
        rel_errors, _, _ = compute_errors(params)
        return np.max(rel_errors)

    def objective_softmax(params, beta=50.0):
        """Soft-max surrogate: log-sum-exp of errors scaled by beta"""
        rel_errors, _, _ = compute_errors(params)
        # log-sum-exp for numerical stability
        max_e = np.max(rel_errors)
        if max_e < 1e-15:
            return max_e
        scaled = beta * rel_errors
        max_scaled = np.max(scaled)
        lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return lse / beta

    def objective_smooth(params):
        """Sum of squared relative errors"""
        rel_errors, _, _ = compute_errors(params)
        return np.sum(rel_errors ** 2)

    def objective_epigraph(params, t):
        """
        Epigraph formulation: minimize t subject to all rel_errors <= t.
        params: 16-dim vector (same as before)
        t: scalar upper bound on max error
        Returns: t + penalty for violations
        """
        rel_errors, _, _ = compute_errors(params)
        # Smooth penalty: sum of softplus(rel_errors - t)
        violations = rel_errors - t
        # Softplus: log(1 + exp(x)) approximates max(0, x)
        penalty = np.sum(np.log1p(np.exp(np.clip(violations, -50, 50))))
        return t + 0.1 * penalty

    # 4. Initialization - Try multiple diverse starting points
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    u0_safe = np.clip(u0, 1e-9, 1 - 1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0.copy()
    best_score = objective_max(p0)
    
    # Try diverse random perturbations to find a better starting point
    rng = np.random.RandomState(42)
    for trial in range(10):
        if time.time() > deadline - 40:
            break
        # Mix of small and medium perturbations
        scale = 0.1 if trial < 5 else 0.3
        perturbation = rng.randn(16) * scale
        p_trial = p0 + perturbation
        # Ensure valid parameters
        theta_trial = np.clip(p_trial[:8], -50, 50)
        alpha_trial = np.clip(p_trial[8:16], -50, 50)
        p_valid = np.concatenate([theta_trial, alpha_trial])
        
        score_trial = objective_max(p_valid)
        if score_trial < best_score:
            best_params = p_valid.copy()
            best_score = score_trial
    
    # 5. Optimization Phase 1: L-BFGS-B on Smooth Objective
    if time.time() < deadline - 35:
        try:
            res1 = scipy.optimize.minimize(
                objective_smooth,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            
            if res1.fun < 1e10:
                score1 = objective_max(res1.x)
                if score1 < best_score:
                    best_params = res1.x.copy()
                    best_score = score1
        except Exception:
            pass

    # 6. Optimization Phase 2: L-BFGS-B on Soft-Max Objective with adaptive beta
    if time.time() < deadline - 25:
        for beta_val in [30.0, 80.0, 200.0, 500.0]:
            if time.time() > deadline - 18:
                break
            try:
                res2 = scipy.optimize.minimize(
                    lambda p, b=beta_val: objective_softmax(p, beta=b),
                    best_params,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                
                if res2.fun < 1e10:
                    score2 = objective_max(res2.x)
                    if score2 < best_score:
                        best_params = res2.x.copy()
                        best_score = score2
            except Exception:
                pass

    # 7. Optimization Phase 3: Epigraph formulation with SLSQP
    if time.time() < deadline - 12:
        try:
            # Initialize t with current best score plus small margin
            t_init = best_score * 1.05 + 1e-6
            
            def epigraph_objective(x):
                params = x[:16]
                t = x[16]
                return objective_epigraph(params, t)
            
            # Bounds: params can be anything reasonable, t >= 0
            bounds = [(-50, 50)] * 16 + [(0, 1.0)]
            
            x_init = np.concatenate([best_params, [t_init]])
            
            res3 = scipy.optimize.minimize(
                epigraph_objective,
                x_init,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 50}
            )
            
            if res3.fun < 1e10:
                params_candidate = res3.x[:16]
                score3 = objective_max(params_candidate)
                if score3 < best_score:
                    best_params = params_candidate.copy()
                    best_score = score3
        except Exception:
            pass

    # 8. Optimization Phase 4: Powell on Max Error Objective
    if time.time() < deadline - 6:
        try:
            res4 = scipy.optimize.minimize(
                objective_max,
                best_params,
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-15, 'ftol': 1e-15}
            )
            
            if res4.fun < best_score:
                best_params = res4.x.copy()
                best_score = res4.fun
        except Exception:
            pass

    # 9. Final Polish: Targeted minimax optimization on worst cases
    if time.time() < deadline - 3:
        # Identify the top-k worst cases for a more targeted optimization
        rel_errors_final, _, _ = compute_errors(best_params)
        top_k = 30
        worst_indices = np.argsort(rel_errors_final)[-top_k:]
        
        # Use a soft-max over just the worst cases for smoother gradient
        def objective_worst_softmax(params, beta=200.0):
            rel_errors, _, _ = compute_errors(params)
            worst_errs = rel_errors[worst_indices]
            max_e = np.max(worst_errs)
            if max_e < 1e-15:
                return max_e
            scaled = beta * worst_errs
            max_scaled = np.max(scaled)
            lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            return lse / beta
        
        try:
            # First pass: soft-max on worst cases
            res5a = scipy.optimize.minimize(
                objective_worst_softmax,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            
            if res5a.fun < 1e10:
                score5a = objective_max(res5a.x)
                if score5a < best_score:
                    best_params = res5a.x.copy()
                    best_score = score5a
            
            # Second pass: Nelder-Mead on hard max of worst cases
            if time.time() < deadline - 2:
                def objective_topk(params):
                    rel_errors, _, _ = compute_errors(params)
                    return np.max(rel_errors[worst_indices])
                
                res5b = scipy.optimize.minimize(
                    objective_topk,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-15, 'fatol': 1e-15}
                )
                
                if res5b.fun < best_score:
                    # Verify with full objective
                    score5b = objective_max(res5b.x)
                    if score5b < best_score:
                        best_params = res5b.x.copy()
                        best_score = score5b
        except Exception:
            pass

    # 10. Final Extraction and Validation
    final_params = best_params
    theta_final = final_params[:8]
    alpha_final = final_params[8:16]
    
    u_final = 1.0 / (1.0 + np.exp(-np.clip(theta_final, -50, 50)))
    
    exp_alpha = np.exp(np.clip(alpha_final, -50, 50) - np.max(np.clip(alpha_final, -50, 50)))
    w_final = exp_alpha / np.sum(exp_alpha)
    
    # Ensure strictly increasing nodes
    sort_idx = np.argsort(u_final)
    u_final = u_final[sort_idx]
    w_final = w_final[sort_idx]
    
    # Clip to [0,1] strictly
    u_final = np.clip(u_final, 1e-12, 1 - 1e-12)
    
    # Ensure weights sum to 1 exactly
    w_final = w_final / np.sum(w_final)
    
    # Check if this is better than pure GL
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u_gl = 0.5 + 0.5 * gl_nodes
    w_gl = 0.5 * gl_weights
    
    t_gl = z_arr[:, None] * u_gl[None, :]
    log_t1 = 3.0 * np.log1p(t_gl)
    log_t2 = (3.0 * (1.0 + w0_arr))[:, None] * np.log1p(t_gl) + \
             (3.0 * w1_arr)[:, None] * (1.0 - 1.0 / (1.0 + t_gl))
    max_log = np.maximum(log_t1, log_t2)
    sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
    log_sum = max_log + np.log(sum_exp)
    f_vals_gl = np.exp(-0.5 * log_sum)
    quad_sums_gl = np.sum(f_vals_gl * w_gl[None, :], axis=1)
    exact_ints_safe = np.maximum(exact_ints, 1e-30)
    gl_err = np.max(np.abs(quad_sums_gl - exact_ints_safe) / exact_ints_safe)
    
    t_opt = z_arr[:, None] * u_final[None, :]
    log_t1 = 3.0 * np.log1p(t_opt)
    log_t2 = (3.0 * (1.0 + w0_arr))[:, None] * np.log1p(t_opt) + \
             (3.0 * w1_arr)[:, None] * (1.0 - 1.0 / (1.0 + t_opt))
    max_log = np.maximum(log_t1, log_t2)
    sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
    log_sum = max_log + np.log(sum_exp)
    f_vals_opt = np.exp(-0.5 * log_sum)
    quad_sums_opt = np.sum(f_vals_opt * w_final[None, :], axis=1)
    opt_err = np.max(np.abs(quad_sums_opt - exact_ints_safe) / exact_ints_safe)
    
    if opt_err > gl_err:
        u_final = u_gl
        w_final = w_gl

    return {"nodes": u_final, "weights": w_final}
