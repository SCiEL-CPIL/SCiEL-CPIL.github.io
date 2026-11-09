"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy:
    1. Start with Gauss-Legendre nodes/weights as initial guess
    2. Optimize nodes and weights jointly to minimize worst-case relative error
       across a representative set of CPL (Chevallier-Polarski-Linder) dark energy models
    3. Use Nelder-Mead optimization with multiple restarts
    4. Ensure positivity of weights and strictly increasing nodes in [0,1]
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline with 10s safety margin
    
    n = 8
    
    # Define CPL models for validation
    # E(a) = H(a)/H0 for flat universe with CPL dark energy
    # E(a) = (a^(-3*(1+w0+wa)) * O_m^(1/3) + a^(-3) * (1-O_m))^(1/2)
    # where O_m is matter density parameter
    # More precisely: E^2(a) = O_m * a^(-3) + O_Lam * a^(-3*(1+w0+wa))
    # with O_m + O_Lam = 1
    
    # Representative set of CPL parameters (w0, wa) and O_m
    # These span a reasonable range of dark energy models
    models = []
    for w0 in [-1.5, -1.2, -1.0, -0.8, -0.5]:
        for wa in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            for Om in [0.25, 0.27, 0.30]:
                models.append((w0, wa, Om))
    
    def E_of_a(a, w0, wa, Om):
        """Compute E(a) = H(a)/H0 for CPL dark energy model."""
        # E^2(a) = Om * a^(-3) + (1-Om) * a^(-3*(1+w0+wa))
        a = np.maximum(a, 1e-15)  # Avoid division by zero
        term_matter = Om * np.power(a, -3.0)
        term_de = (1.0 - Om) * np.power(a, -3.0 * (1.0 + w0 + wa))
        return np.sqrt(term_matter + term_de)
    
    def integrand(z, w0, wa, Om):
        """Compute the integrand for luminosity distance: z * (1+z) / E(a)"""
        a = 1.0 / (1.0 + z)
        E = E_of_a(a, w0, wa, Om)
        return z * (1.0 + z) / E
    
    def luminosity_distance_exact(z, w0, wa, Om, n_points=10000):
        """Compute exact luminosity distance using adaptive Simpson integration."""
        if z <= 0:
            return 0.0
        # Use scipy's quad for adaptive integration
        from scipy.integrate import quad
        result, _ = quad(integrand, 0, z, args=(w0, wa, Om), limit=200)
        return result
    
    def quadrature_error(nodes, weights, w0, wa, Om, z_eval=1.0):
        """Compute relative error of quadrature vs exact for a given model and redshift."""
        a_nodes = 1.0 / (1.0 + z_eval * nodes)
        E_nodes = E_of_a(a_nodes, w0, wa, Om)
        quad_result = z_eval * (1.0 + z_eval) * np.sum(weights / E_nodes)
        exact_result = luminosity_distance_exact(z_eval, w0, wa, Om)
        if abs(exact_result) < 1e-15:
            return 0.0
        return abs(quad_result - exact_result) / abs(exact_result) * 100.0
    
    def worst_case_error(nodes, weights, z_evals=None):
        """Compute worst-case relative error across all models and redshifts."""
        if z_evals is None:
            z_evals = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        max_err = 0.0
        for (w0, wa, Om) in models:
            for z in z_evals:
                err = quadrature_error(nodes, weights, w0, wa, Om, z)
                max_err = max(max_err, err)
        return max_err
    
    def objective(params):
        """Optimization objective: worst-case error with penalty for constraint violations."""
        # params: [node1, node2, ..., node8, weight1, weight2, ..., weight8]
        # But we need to enforce constraints. Let's use a transformation.
        # Actually, let's optimize over unconstrained parameters and transform.
        # For now, let's work with raw nodes and weights and add penalties.
        
        nodes = params[:n]
        weights = params[n:]
        
        # Penalty for nodes not in (0,1) or not strictly increasing
        penalty = 0.0
        for i in range(n):
            if nodes[i] <= 0 or nodes[i] >= 1:
                penalty += 1000.0
        for i in range(n - 1):
            if nodes[i] >= nodes[i + 1]:
                penalty += 1000.0
        
        # Penalty for negative weights
        for i in range(n):
            if weights[i] < 0:
                penalty += 1000.0
        
        # Penalty for weights not summing to 1
        penalty += 100.0 * abs(np.sum(weights) - 1.0)
        
        # Compute actual error
        err = worst_case_error(nodes, weights)
        
        return err + penalty
    
    # Initial guess: Gauss-Legendre
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(n)
    nodes_init = (x_gl + 1.0) / 2.0
    weights_init = w_gl / 2.0
    
    # Convert to flat parameter vector
    params_init = np.concatenate([nodes_init, weights_init])
    
    # Multi-start Nelder-Mead optimization
    best_params = params_init.copy()
    best_error = objective(params_init)
    
    n_restarts = 10
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate perturbed initial guess
        if restart == 0:
            params_current = params_init.copy()
        else:
            # Perturb nodes slightly
            node_perturb = np.random.uniform(-0.05, 0.05, n)
            weight_perturb = np.random.uniform(-0.05, 0.05, n)
            params_current = params_init.copy()
            params_current[:n] += node_perturb
            params_current[n:] += weight_perturb
            # Normalize weights to sum to 1
            params_current[n:] /= np.sum(params_current[n:])
        
        # Run Nelder-Mead
        try:
            result = minimize(objective, params_current, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    # Extract best nodes and weights
    nodes_best = best_params[:n]
    weights_best = best_params[n:]
    
    # Ensure constraints are satisfied
    # Sort nodes and corresponding weights
    sort_idx = np.argsort(nodes_best)
    nodes_best = nodes_best[sort_idx]
    weights_best = weights_best[sort_idx]
    
    # Clamp nodes to (0,1) with small margin
    eps = 1e-10
    nodes_best = np.clip(nodes_best, eps, 1.0 - eps)
    
    # Ensure strictly increasing by adding small increments if needed
    for i in range(1, n):
        if nodes_best[i] <= nodes_best[i-1]:
            nodes_best[i] = nodes_best[i-1] + 1e-8
    
    # Ensure nodes are in (0,1)
    nodes_best = np.clip(nodes_best, eps, 1.0 - eps)
    
    # Ensure weights are positive and sum to 1
    weights_best = np.maximum(weights_best, 1e-15)
    weights_best = weights_best / np.sum(weights_best)
    
    # Final check: ensure strictly increasing nodes
    for i in range(1, n):
        if nodes_best[i] <= nodes_best[i-1]:
            nodes_best[i] = nodes_best[i-1] + 1e-8
    
    # Final clamp
    nodes_best = np.clip(nodes_best, eps, 1.0 - eps)
    
    return {"nodes": nodes_best, "weights": weights_best}
# EVOLVE-BLOCK-END
