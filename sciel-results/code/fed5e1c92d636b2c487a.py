import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    # Define a grid of cosmological parameters for validation
    w0_vals = np.linspace(-1.2, 0.5, 12)
    w1_vals = np.linspace(-1.5, 1.5, 12)
    
    # Define a grid of redshifts for evaluation
    z_vals = np.concatenate([
        np.linspace(0.02, 0.15, 8),
        np.linspace(0.2, 1.0, 10),
        np.linspace(1.5, 3.0, 10),
        np.linspace(3.5, 6.0, 5)
    ])
    
    # Precompute exact integrals and integrand values at candidate nodes
    # We'll compute exact integrals using quad
    exact_ints = []
    model_params = []
    
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                def integrand(u, w0=w0, w1=w1, z=z):
                    t = z * u
                    log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
                    return np.exp(-log_E)
                
                try:
                    val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))
                except Exception:
                    us = np.linspace(0, 1, 10001)
                    ts = z * us
                    log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
                    vals = np.exp(-log_E)
                    val = np.trapz(vals, us)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))

    exact_ints = np.array(exact_ints)
    n_models = len(exact_ints)
    
    w0_arr = np.array([p[0] for p in model_params])
    w1_arr = np.array([p[1] for p in model_params])
    z_arr = np.array([p[2] for p in model_params])
    
    def compute_integrand_matrix(nodes):
        """Compute f(t_i) for all models and nodes. Shape: (n_models, 8)"""
        t = z_arr[:, None] * nodes[None, :]
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        return f_vals
    
    def solve_weights_linear(nodes):
        """
        Given fixed nodes, solve for optimal weights via linear programming.
        Minimize epsilon subject to:
          -epsilon <= sum_j w_j f_ij - exact_i <= epsilon for all i
          sum_j w_j = 1
          w_j >= 0
        """
        f_mat = compute_integrand_matrix(nodes)  # (n_models, 8)
        n_nodes = 8
        
        # Variables: w_0, ..., w_7, epsilon
        # Minimize: epsilon
        # Constraints:
        #   sum_j f_ij * w_j - exact_i <= epsilon  for all i
        #   -sum_j f_ij * w_j + exact_i <= epsilon  for all i
        #   sum_j w_j = 1
        #   w_j >= 0, epsilon >= 0
        
        c = np.zeros(n_nodes + 1)
        c[n_nodes] = 1.0  # minimize epsilon
        
        # Inequality constraints: A_ub @ x <= b_ub
        # For each model i:
        #   sum_j f_ij * w_j - epsilon <= exact_i
        #  -sum_j f_ij * w_j - epsilon <= -exact_i
        
        A_ub = np.zeros((2 * n_models, n_nodes + 1))
        b_ub = np.zeros(2 * n_models)
        
        for i in range(n_models):
            # Row 2i: sum_j f_ij * w_j - epsilon <= exact_i
            A_ub[2*i, :n_nodes] = f_mat[i, :]
            A_ub[2*i, n_nodes] = -1.0
            b_ub[2*i] = exact_ints[i]
            
            # Row 2i+1: -sum_j f_ij * w_j - epsilon <= -exact_i
            A_ub[2*i+1, :n_nodes] = -f_mat[i, :]
            A_ub[2*i+1, n_nodes] = -1.0
            b_ub[2*i+1] = -exact_ints[i]
        
        # Equality constraint: sum_j w_j = 1
        A_eq = np.zeros((1, n_nodes + 1))
        A_eq[0, :n_nodes] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w_j >= 0, epsilon >= 0
        bounds = [(0, None)] * n_nodes + [(0, None)]
        
        try:
            result = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                                            bounds=bounds, method='highs')
            if result.success:
                weights = result.x[:n_nodes]
                # Ensure weights are non-negative and sum to 1
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)
                return weights, result.x[n_nodes]
            else:
                # Fallback: use Gauss-Legendre weights
                _, gl_w = np.polynomial.legendre.leggauss(8)
                gl_w = 0.5 * gl_w
                return gl_w, None
        except Exception:
            _, gl_w = np.polynomial.legendre.leggauss(8)
            gl_w = 0.5 * gl_w
            return gl_w, None
    
    def evaluate_nodes(nodes):
        """Evaluate the worst-case relative error for given nodes (with optimal weights)."""
        nodes = np.sort(nodes)
        weights, eps = solve_weights_linear(nodes)
        if weights is None:
            return np.inf
        
        f_mat = compute_integrand_matrix(nodes)
        quad_sums = f_mat @ weights
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        return np.max(rel_errors)
    
    # Strategy: 
    # 1. Coarse grid search over node configurations
    # 2. Local refinement using Nelder-Mead on node positions
    # 3. Multiple restarts from different initial points
    
    # Initial candidate: Gauss-Legendre nodes on [0,1]
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    gl_nodes_01 = 0.5 + 0.5 * gl_nodes
    
    best_nodes = gl_nodes_01.copy()
    best_error = evaluate_nodes(best_nodes)
    
    # Coarse search: perturb Gauss-Legendre nodes
    def perturb_gl(scale, rng):
        """Generate perturbed Gauss-Legendre nodes."""
        nodes = gl_nodes_01 + scale * rng.standard_normal(8)
        # Ensure nodes are in (0,1) and sorted
        nodes = np.clip(nodes, 0.01, 0.99)
        nodes = np.sort(nodes)
        # Ensure strict increase
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
        return nodes
    
    # Try several scales
    rng = np.random.RandomState(42)
    scales = [0.1, 0.05, 0.02, 0.01]
    
    for scale in scales:
        for _ in range(20):
            if time.time() - start_time > time_limit - 15:
                break
            
            candidate = perturb_gl(scale, rng)
            err = evaluate_nodes(candidate)
            if err < best_error:
                best_error = err
                best_nodes = candidate.copy()
    
    # Local refinement using Nelder-Mead on node positions
    # Parameterize nodes to ensure they stay in (0,1) and sorted
    def nodes_to_params(nodes):
        """Transform nodes to unconstrained parameters using logit-like transform."""
        # Use a transform that ensures nodes are in (0,1)
        # u_i = 1/(1+exp(-x_i)), but we need to maintain ordering
        # Simpler: just clip and use the raw values with bounds
        return nodes
    
    def params_to_nodes(params):
        """Transform params to valid nodes."""
        nodes = np.clip(params, 0.005, 0.995)
        # Ensure strict increase
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-7
        return nodes
    
    def objective_nodes(params):
        nodes = params_to_nodes(params)
        return evaluate_nodes(nodes)
    
    # Nelder-Mead refinement
    if time.time() - start_time < time_limit - 10:
        try:
            res = scipy.optimize.minimize(
                objective_nodes,
                best_nodes,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_error:
                best_error = res.fun
                best_nodes = params_to_nodes(res.x)
        except Exception:
            pass
    
    # Additional restarts from different initial configurations
    if time.time() - start_time < time_limit - 10:
        # Try Chebyshev-like nodes
        cheb_nodes = 0.5 - 0.5 * np.cos(np.pi * (np.arange(8) + 0.5) / 8)
        err = evaluate_nodes(cheb_nodes)
        if err < best_error:
            best_error = err
            best_nodes = cheb_nodes.copy()
        
        # Try equally spaced nodes
        eq_nodes = np.linspace(0.05, 0.95, 8)
        err = evaluate_nodes(eq_nodes)
        if err < best_error:
            best_error = err
            best_nodes = eq_nodes.copy()
        
        # Refine from these if they're better
        if time.time() - start_time < time_limit - 5:
            try:
                res = scipy.optimize.minimize(
                    objective_nodes,
                    best_nodes,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_error:
                    best_error = res.fun
                    best_nodes = params_to_nodes(res.x)
            except Exception:
                pass
    
    # Final: get optimal weights for best nodes
    weights, _ = solve_weights_linear(best_nodes)
    
    # Ensure weights are positive and sum to 1
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing in [0,1]
    best_nodes = np.sort(best_nodes)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-10
    best_nodes = np.clip(best_nodes, 1e-10, 1-1e-10)
    
    return {"nodes": best_nodes, "weights": weights}
