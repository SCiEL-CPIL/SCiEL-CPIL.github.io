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

    # 4. Initialization
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
    
    # 5. Optimization Phase 1: L-BFGS-B on Smooth Objective
    if time.time() < deadline - 15:
        try:
            res1 = scipy.optimize.minimize(
                objective_smooth,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res1.fun < 1e10:
                score1 = objective_max(res1.x)
                if score1 < best_score:
                    best_params = res1.x.copy()
                    best_score = score1
        except Exception:
            pass

    # 6. Optimization Phase 2: L-BFGS-B on Soft-Max Objective
    if time.time() < deadline - 10:
        try:
            res2 = scipy.optimize.minimize(
                objective_softmax,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res2.fun < 1e10:
                score2 = objective_max(res2.x)
                if score2 < best_score:
                    best_params = res2.x.copy()
                    best_score = score2
        except Exception:
            pass

    # 7. Optimization Phase 3: Powell on Max Error Objective
    if time.time() < deadline - 5:
        try:
            res3 = scipy.optimize.minimize(
                objective_max,
                best_params,
                method='Powell',
                options={'maxiter': 150, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            
            if res3.fun < best_score:
                best_params = res3.x.copy()
                best_score = res3.fun
        except Exception:
            pass

    # 8. Final Polish: Nelder-Mead with tight tolerance
    if time.time() < deadline - 2:
        try:
            res4 = scipy.optimize.minimize(
                objective_max,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            
            if res4.fun < best_score:
                best_params = res4.x.copy()
                best_score = res4.fun
        except Exception:
            pass

    # 9. Final Extraction and Validation
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
