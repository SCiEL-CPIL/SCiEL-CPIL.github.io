import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s buffer for 90s limit

    # Define a diverse set of validation cosmologies (w0, w1, z)
    # Covering the parameter space of interest
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
    w1_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    z_vals = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    # Create a representative subset to keep computation manageable
    # We want good coverage but not too many points for quad
    validation_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                validation_cases.append((w0, w1, z))
    
    # Subsample to keep it to ~60 cases for speed
    n_total = len(validation_cases)
    n_target = 60
    if n_total > n_target:
        idx = np.linspace(0, n_total - 1, n_target).astype(int)
        validation_cases = [validation_cases[i] for i in idx]
    
    def integrand(u, w0, w1, z):
        t = z * u
        # E(t)^2 = (1+t)^3 + (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
        # 1/E(t) = 1 / sqrt((1+t)^3 + (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t))))
        # Use log-space for stability
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_term1, log_term2)
        log_sum = max_log + np.log(np.exp(log_term1 - max_log) + np.exp(log_term2 - max_log))
        
        return np.exp(-0.5 * log_sum)

    # Precompute exact integrals for the validation set
    exact_ints = []
    valid_cases = []
    for w0, w1, z in validation_cases:
        try:
            val, _ = scipy.integrate.quad(lambda u: integrand(u, w0, w1, z), 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
            exact_ints.append(val)
            valid_cases.append((w0, w1, z))
        except Exception:
            # Fallback to simple trapezoidal if quad fails
            us = np.linspace(0, 1, 1001)
            vals = np.array([integrand(u, w0, w1, z) for u in us])
            val = np.trapz(vals, us)
            exact_ints.append(val)
            valid_cases.append((w0, w1, z))
            
    exact_ints = np.array(exact_ints)
    w0_arr = np.array([c[0] for c in valid_cases])
    w1_arr = np.array([c[1] for c in valid_cases])
    z_arr = np.array([c[2] for c in valid_cases])
    n_cases = len(valid_cases)

    # Compute integrand values for a given set of nodes
    def compute_integrand_values(nodes):
        t = z_arr[:, None] * nodes[None, :]
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        max_log = np.maximum(log_term1, log_term2)
        log_sum = max_log + np.log(np.exp(log_term1 - max_log) + np.exp(log_term2 - max_log))
        return np.exp(-0.5 * log_sum)

    # Objective function: Max relative error
    def objective(params):
        u = params[:8]
        w = params[8:16]
        
        if np.any(u <= 0) or np.any(u >= 1) or np.any(w <= 0):
            return 1e10
        if np.any(np.diff(u) <= 0):
            return 1e10
        if abs(np.sum(w) - 1.0) > 1e-6:
            return 1e10
            
        f_vals = compute_integrand_values(u)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        return np.max(rel_errors)

    # Smooth surrogate objective: sum of squared log-errors
    def objective_smooth(params):
        u = params[:8]
        w = params[8:16]
        
        if np.any(u <= 0) or np.any(u >= 1) or np.any(w <= 0):
            return 1e10
        if np.any(np.diff(u) <= 0):
            return 1e10
        if abs(np.sum(w) - 1.0) > 1e-6:
            return 1e10
            
        f_vals = compute_integrand_values(u)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Avoid log of zero or negative
        quad_sums_safe = np.maximum(quad_sums, 1e-30)
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        
        log_errors = np.log(quad_sums_safe) - np.log(exact_ints_safe)
        return np.sum(log_errors ** 2)

    # Initial guess: Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    p0 = np.concatenate([u0, w0_init])
    
    best_params = p0.copy()
    best_score = objective(p0)
    
    # Phase 1: L-BFGS-B on smooth surrogate
    if time.time() < deadline - 20:
        try:
            # Bounds: nodes in (1e-6, 1-1e-6), weights in (1e-12, 1)
            bounds = [(1e-6, 1 - 1e-6)] * 8 + [(1e-12, 1.0)] * 8
            
            res1 = scipy.optimize.minimize(
                objective_smooth,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res1.fun < 1e6:  # Valid result
                score1 = objective(res1.x)
                if score1 < best_score:
                    best_params = res1.x.copy()
                    best_score = score1
        except Exception:
            pass

    # Phase 2: Powell refinement on max-error objective
    if time.time() < deadline - 10:
        try:
            bounds = [(1e-6, 1 - 1e-6)] * 8 + [(1e-12, 1.0)] * 8
            
            res2 = scipy.optimize.minimize(
                objective,
                best_params,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            
            if res2.fun < best_score:
                best_params = res2.x.copy()
                best_score = res2.fun
        except Exception:
            pass

    # Phase 3: Nelder-Mead for further refinement
    if time.time() < deadline - 5:
        try:
            res3 = scipy.optimize.minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res3.fun < best_score:
                best_params = res3.x.copy()
                best_score = res3.fun
        except Exception:
            pass

    # Final cleanup and validation
    u_final = best_params[:8]
    w_final = best_params[8:16]
    
    # Ensure nodes are strictly increasing and in (0,1)
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    
    # Clip nodes to [1e-6, 1-1e-6]
    u_sorted = np.clip(u_sorted, 1e-6, 1 - 1e-6)
    
    # Ensure weights are positive and sum to 1
    w_sorted = np.clip(w_sorted, 1e-12, None)
    w_norm = w_sorted / np.sum(w_sorted)
    
    # Final check: compare with pure GL
    f_vals_final = compute_integrand_values(u_sorted)
    quad_sums_final = np.sum(f_vals_final * w_norm[None, :], axis=1)
    final_err = np.max(np.abs(quad_sums_final - exact_ints) / exact_ints)
    
    f_vals_gl = compute_integrand_values(u0)
    quad_sums_gl = np.sum(f_vals_gl * w0_init[None, :], axis=1)
    gl_err = np.max(np.abs(quad_sums_gl - exact_ints) / exact_ints)
    
    if final_err > gl_err:
        u_sorted = u0.copy()
        w_norm = w0_init.copy()
    
    return {"nodes": u_sorted, "weights": w_norm}
