import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer

    # Define a small, diverse set of validation cosmologies
    # (w0, w1, z)
    # We pick points that cover the range of typical CPL parameters
    w0_vals = [-1.0, -0.5, 0.0, 0.5]
    w1_vals = [-1.0, 0.0, 1.0]
    z_vals = [0.5, 1.0, 2.0, 5.0]
    
    # Create a subset of combinations to keep computation fast
    # 4 * 3 * 4 = 48 is too many for quad if we want to be safe, let's pick 12 representative ones
    validation_cases = [
        (-1.0, -1.0, 0.5),
        (-1.0, 0.0, 1.0),
        (-1.0, 1.0, 2.0),
        (-0.5, -1.0, 0.5),
        (-0.5, 0.0, 1.5),
        (-0.5, 1.0, 3.0),
        (0.0, -1.0, 0.5),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 2.5),
        (0.5, -1.0, 0.5),
        (0.5, 0.0, 1.0),
        (0.5, 1.0, 2.0),
        (-1.0, 0.0, 5.0),
        (0.0, 0.0, 5.0),
        (1.0, -1.0, 0.5), # Extreme w0
        (-1.5, 0.0, 1.0), # Extreme w0
    ]
    
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
            # Fallback to simple trapezoidal if quad fails (unlikely)
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

    # Objective function: Max relative error
    def objective(params):
        # params: 8 nodes, 8 weights
        # We assume nodes are already sorted and in (0,1), weights sum to 1
        # But during optimization, we might need to enforce this.
        # For Nelder-Mead, we can just pass raw values and handle constraints in the function?
        # No, it's better to parameterize or just check validity.
        # Let's use a parameterization that guarantees constraints.
        # Nodes: 7 positive deltas + 1 start pos
        # Weights: 7 positive weights + 1 remaining
        
        # Actually, let's just use raw nodes/weights and penalize invalid ones heavily.
        # Or, simpler: Start with GL, which is valid. Use Nelder-Mead which stays in a region.
        
        u = params[:8]
        w = params[8:16]
        
        # Check validity
        if np.any(u <= 0) or np.any(u >= 1) or np.any(w <= 0):
            return 1e10
        if np.any(np.diff(u) <= 0):
            return 1e10
        if abs(np.sum(w) - 1.0) > 1e-6:
            return 1e10
            
        # Calculate quadrature sums
        t = z_arr[:, None] * u[None, :]
        
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_term1, log_term2)
        log_sum = max_log + np.log(np.exp(log_term1 - max_log) + np.exp(log_term2 - max_log))
        f_vals = np.exp(-0.5 * log_sum)
        
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        return np.max(rel_errors)

    # Initial guess: Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    p0 = np.concatenate([u0, w0_init])
    
    best_params = p0
    best_score = objective(p0)
    
    # Optimization
    if time.time() < deadline - 10:
        try:
            # Nelder-Mead is good for small problems and doesn't require gradients
            # It can handle the non-smoothness of max() well
            res = scipy.optimize.minimize(
                objective, 
                p0, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass

    # Final cleanup and validation
    u_final = best_params[:8]
    w_final = best_params[8:16]
    
    # Ensure nodes are strictly increasing and in (0,1)
    # Sort them just in case, and re-sort weights accordingly
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    
    # Clip nodes to [1e-6, 1-1e-6] to be strictly inside
    u_sorted = np.clip(u_sorted, 1e-6, 1 - 1e-6)
    
    # Ensure weights are positive and sum to 1
    w_sorted = np.clip(w_sorted, 1e-12, None)
    w_norm = w_sorted / np.sum(w_sorted)
    
    # Final check: if the optimized result is somehow worse or invalid, fallback to GL
    # Calculate error for the final result
    t = z_arr[:, None] * u_sorted[None, :]
    log_term1 = 3.0 * np.log1p(t)
    log_term2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
    max_log = np.maximum(log_term1, log_term2)
    log_sum = max_log + np.log(np.exp(log_term1 - max_log) + np.exp(log_term2 - max_log))
    f_vals = np.exp(-0.5 * log_sum)
    quad_sums = np.sum(f_vals * w_norm[None, :], axis=1)
    final_err = np.max(np.abs(quad_sums - exact_ints) / exact_ints)
    
    # Calculate error for pure GL
    t_gl = z_arr[:, None] * u0[None, :]
    log_term1_gl = 3.0 * np.log1p(t_gl)
    log_term2_gl = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t_gl) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t_gl))
    max_log_gl = np.maximum(log_term1_gl, log_term2_gl)
    log_sum_gl = max_log_gl + np.log(np.exp(log_term1_gl - max_log_gl) + np.exp(log_term2_gl - max_log_gl))
    f_vals_gl = np.exp(-0.5 * log_sum_gl)
    quad_sums_gl = np.sum(f_vals_gl * w0_init[None, :], axis=1)
    gl_err = np.max(np.abs(quad_sums_gl - exact_ints) / exact_ints)
    
    if final_err > gl_err:
        u_sorted = u0
        w_norm = w0_init
    
    return {"nodes": u_sorted, "weights": w_norm}
