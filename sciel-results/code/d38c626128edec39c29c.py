import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Define a diverse set of validation cosmologies
    # We use a mix of standard and extreme CPL parameters
    # The key is to cover the range of integrand shapes (curvature, steepness)
    w0_vals = [-1.2, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
    w1_vals = [-1.5, -0.8, -0.3, 0.0, 0.3, 0.8, 1.5]
    z_vals = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    
    # We cannot test all 8*7*7 = 392 combinations with quad in 85s if we want to be safe.
    # Let's select a representative subset for the optimization objective.
    # We will use a stratified sampling approach.
    
    # Actually, let's just pick a fixed set of ~30-50 cases that are well distributed.
    # This avoids the overhead of generating a huge grid and computing exact integrals for all of them.
    
    validation_cases = [
        # (w0, w1, z)
        (-1.2, -1.5, 0.1), (-1.2, 0.0, 0.5), (-1.2, 1.5, 1.0),
        (-0.8, -1.5, 0.5), (-0.8, 0.0, 1.0), (-0.8, 1.5, 2.0),
        (-0.5, -1.5, 1.0), (-0.5, 0.0, 2.0), (-0.5, 1.5, 3.0),
        (-0.2, -1.5, 2.0), (-0.2, 0.0, 3.0), (-0.2, 1.5, 5.0),
        (0.0, -1.5, 3.0), (0.0, 0.0, 5.0), (0.0, 1.5, 8.0),
        (0.2, -1.5, 5.0), (0.2, 0.0, 8.0), (0.2, 1.5, 0.1),
        (0.5, -1.5, 8.0), (0.5, 0.0, 0.1), (0.5, 1.5, 0.5),
        (0.8, -1.5, 0.1), (0.8, 0.0, 0.5), (0.8, 1.5, 1.0),
        # Add some mid-range for stability
        (-0.5, -0.5, 0.5), (-0.5, 0.5, 1.5),
        (0.0, -0.5, 1.5), (0.0, 0.5, 2.5),
        (0.5, -0.5, 2.5), (0.5, 0.5, 4.0),
        (-1.0, -1.0, 1.0), (1.0, 1.0, 1.0),
    ]

    def integrand(u, w0, w1, z):
        t = z * u
        # E(t)^2 = (1+t)^3 + (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
        # 1/E(t) = 1 / sqrt(...)
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
            # Use high precision quad
            val, _ = scipy.integrate.quad(lambda u: integrand(u, w0, w1, z), 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
            exact_ints.append(val)
            valid_cases.append((w0, w1, z))
        except Exception:
            # Fallback
            us = np.linspace(0, 1, 2001)
            vals = np.array([integrand(u, w0, w1, z) for u in us])
            val = np.trapz(vals, us)
            exact_ints.append(val)
            valid_cases.append((w0, w1, z))
            
    exact_ints = np.array(exact_ints)
    w0_arr = np.array([c[0] for c in valid_cases])
    w1_arr = np.array([c[1] for c in valid_cases])
    z_arr = np.array([c[2] for c in valid_cases])
    n_cases = len(valid_cases)

    # Parameterization:
    # Nodes: 8 values in (0,1), strictly increasing.
    # Weights: 8 positive values summing to 1.
    # To handle constraints robustly in unconstrained optimizers (like Nelder-Mead or Powell),
    # we can use a soft parameterization or just penalize invalid solutions.
    # Given the small dimensionality (16 params), Nelder-Mead with penalties is often effective.
    # However, the "max error" objective is non-smooth.
    
    # Let's try a different structural approach:
    # Instead of optimizing nodes and weights directly, we can use a "hybrid" approach.
    # 1. Start with Gauss-Legendre.
    # 2. Use Nelder-Mead to optimize nodes and weights with a penalty for constraint violation.
    # 3. Refine with a local search.

    def compute_errors(nodes, weights):
        """
        Compute relative errors for the given nodes and weights.
        Returns max relative error.
        """
        # Ensure nodes are sorted
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weights = weights[sort_idx]
        
        # Check constraints
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e10
            
        # Calculate quadrature sums
        # t = z * u
        t = z_arr[:, None] * nodes[None, :]
        
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = 3.0 * (w0_arr[:, None] + 1.0) * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_term1, log_term2)
        log_sum = max_log + np.log(np.exp(log_term1 - max_log) + np.exp(log_term2 - max_log))
        f_vals = np.exp(-0.5 * log_sum)
        
        quad_sums = np.sum(f_vals * weights[None, :], axis=1)
        
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        return np.max(rel_errors)

    # Initial guess: Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    p0 = np.concatenate([u0, w0_init])
    
    best_params = p0
    best_score = compute_errors(u0, w0_init)
    
    # Optimization Strategy:
    # 1. Nelder-Mead on the raw max error with penalties.
    # 2. If time permits, try a second pass with a different initial guess or method.
    
    def objective(params):
        u = params[:8]
        w = params[8:16]
        return compute_errors(u, w)

    # Run Nelder-Mead
    if time.time() < deadline - 15:
        try:
            res = scipy.optimize.minimize(
                objective, 
                p0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass

    # Try a second optimization starting from a slightly perturbed initial guess
    # to avoid local minima, if time allows.
    if time.time() < deadline - 10:
        # Perturb GL nodes slightly
        u1 = u0 + 0.05 * np.random.randn(8)
        u1 = np.clip(u1, 0.01, 0.99)
        # Sort to ensure increasing order for the penalty function
        u1_sorted = np.sort(u1)
        w1 = w0_init + 0.05 * np.random.randn(8)
        w1 = np.clip(w1, 1e-4, None)
        w1 = w1 / np.sum(w1)
        
        p1 = np.concatenate([u1_sorted, w1])
        
        try:
            res2 = scipy.optimize.minimize(
                objective, 
                p1, 
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
        except Exception:
            pass

    # Final cleanup and validation
    u_final = best_params[:8]
    w_final = best_params[8:16]
    
    # Ensure nodes are strictly increasing and in (0,1)
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
    final_err = compute_errors(u_sorted, w_norm)
    
    # Calculate error for pure GL
    gl_err = compute_errors(u0, w0_init)
    
    if final_err > gl_err or not np.isfinite(final_err):
        u_sorted = u0
        w_norm = w0_init
    
    return {"nodes": u_sorted, "weights": w_norm}
