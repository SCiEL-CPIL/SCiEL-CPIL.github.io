import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Define Validation Cosmologies
    # Use a moderately dense grid to balance coverage and computation time.
    # The key insight is that the integrand f(u) = 1/E(z*u) has varying shape
    # depending on w0, w1, and z. We want to capture the worst-case error.
    
    # Redshifts: Log-spaced for low z, linear for high z
    # More points at low z where the function is smoother but the integral is small,
    # and more points at high z where the function changes rapidly.
    z_vals = np.unique(np.concatenate([
        np.logspace(np.log10(0.01), np.log10(1.0), 15),
        np.linspace(1.0, 10.0, 25)
    ]))
    
    # w0 and w1 ranges: Standard CPL parameter space for validation
    w0_vals = np.linspace(-1.5, 0.5, 14)
    w1_vals = np.linspace(-1.5, 1.5, 14)
    
    validation_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                validation_cases.append((w0, w1, z))
                
    n_cases = len(validation_cases)
    
    # 2. Precompute Exact Integrals using vectorized Gauss-Legendre
    # This is significantly faster than calling scipy.integrate.quad n_cases times.
    # We use a high-order Gauss-Legendre rule (e.g., 128 points) to get a very accurate
    # reference value for the integral.
    
    # Precompute 128-point Gauss-Legendre on [0,1]
    gl_x, gl_w = np.polynomial.legendre.leggauss(128)
    gl_x = 0.5 + 0.5 * gl_x  # Map to [0,1]
    gl_w = 0.5 * gl_w        # Scale weights
    
    w0_arr = np.zeros(n_cases)
    w1_arr = np.zeros(n_cases)
    z_arr = np.zeros(n_cases)
    
    for i, (w0, w1, z) in enumerate(validation_cases):
        w0_arr[i] = w0
        w1_arr[i] = w1
        z_arr[i] = z
    
    # Vectorized computation of integrand values
    # t shape: (n_cases, 128)
    t = z_arr[:, None] * gl_x[None, :]
    
    # Compute E(t)^2 = (1+t)^3 + (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
    # We work in log-space for stability
    log_t1 = 3.0 * np.log1p(t)
    log_t2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
    
    max_log = np.maximum(log_t1, log_t2)
    sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
    log_sum = max_log + np.log(sum_exp)
    
    f_vals = np.exp(-0.5 * log_sum)  # shape (n_cases, 128)
    
    # Exact integrals via Gauss-Legendre
    exact_ints = np.sum(f_vals * gl_w[None, :], axis=1)
    
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
        
        # Nodes in (0, 1)
        theta_clipped = np.clip(theta, -50, 50)
        u = 1.0 / (1.0 + np.exp(-theta_clipped))
        
        # Weights positive, sum to 1
        alpha_clipped = np.clip(alpha, -50, 50)
        exp_alpha = np.exp(alpha_clipped - np.max(alpha_clipped))
        w = exp_alpha / np.sum(exp_alpha)
        
        # Vectorized computation of f_vals
        t = z_arr[:, None] * u[None, :]
        
        log_t1 = 3.0 * np.log1p(t)
        log_t2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_t1, log_t2)
        sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
        log_sum = max_log + np.log(sum_exp)
        
        f_vals = np.exp(-0.5 * log_sum)  # shape (n_cases, 8)
        
        # Quadrature sum: sum_i w_i f_i
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Relative errors
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        rel_errors = np.abs(quad_sums - exact_ints_safe) / exact_ints_safe
        
        return rel_errors, u, w

    def objective_max(params):
        """Max relative error"""
        rel_errors, _, _ = compute_errors(params)
        return np.max(rel_errors)

    def objective_smooth(params):
        """
        Smooth surrogate: Sum of squared relative errors
        """
        rel_errors, _, _ = compute_errors(params)
        return np.sum(rel_errors ** 2)
        
    def objective_soft_max(params):
        """
        Soft-max of relative errors, approximating max but differentiable.
        This helps L-BFGS-B navigate the non-smooth max landscape.
        """
        rel_errors, _, _ = compute_errors(params)
        # Scale to avoid overflow, temperature controls smoothness
        # A lower temperature makes it closer to max.
        # We want to penalize large errors strongly.
        scaled = rel_errors * 100.0 
        max_scaled = np.max(scaled)
        # If max is very small, it's just a scaled max.
        # If large, it smooths the peak.
        # Actually, for minimization, we can just use the log-sum-exp trick on scaled errors.
        # But we need to be careful with the scale.
        # Let's stick to sum of squares for L-BFGS and use Powell/Nelder-Mead for max.
        # The inspiration suggests using a smooth surrogate.
        # Let's try a different smooth surrogate: sum of (rel_errors + epsilon)^p
        # But sum of squares is standard.
        return np.sum(rel_errors ** 2)

    # 4. Initialization
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    # Map to parameter space
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
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res1.fun < 1e10:
                score1 = objective_max(res1.x)
                if score1 < best_score:
                    best_params = res1.x.copy()
                    best_score = score1
        except Exception:
            pass

    # 6. Optimization Phase 2: Nelder-Mead on Max Error Objective
    # Nelder-Mead is robust for non-smooth functions like max error.
    if time.time() < deadline - 5:
        try:
            res2 = scipy.optimize.minimize(
                objective_max,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if res2.fun < best_score:
                best_params = res2.x.copy()
                best_score = res2.fun
        except Exception:
            pass

    # 7. Optimization Phase 3: Powell on Max Error (if time permits)
    # Powell is derivative-free and often good for final polishing.
    if time.time() < deadline - 2:
        try:
            res3 = scipy.optimize.minimize(
                objective_max,
                best_params,
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            
            if res3.fun < best_score:
                best_params = res3.x.copy()
                best_score = res3.fun
        except Exception:
            pass

    # 8. Final Extraction and Validation
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
    
    # Compute error for GL
    t_gl = z_arr[:, None] * u_gl[None, :]
    log_t1 = 3.0 * np.log1p(t_gl)
    log_t2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t_gl) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t_gl))
    max_log = np.maximum(log_t1, log_t2)
    sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
    log_sum = max_log + np.log(sum_exp)
    f_vals_gl = np.exp(-0.5 * log_sum)
    quad_sums_gl = np.sum(f_vals_gl * w_gl[None, :], axis=1)
    exact_ints_safe = np.maximum(exact_ints, 1e-30)
    gl_err = np.max(np.abs(quad_sums_gl - exact_ints_safe) / exact_ints_safe)
    
    # Compute error for optimized
    t_opt = z_arr[:, None] * u_final[None, :]
    log_t1 = 3.0 * np.log1p(t_opt)
    log_t2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t_opt) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t_opt))
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
