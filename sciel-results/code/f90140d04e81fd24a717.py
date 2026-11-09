import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Broader set of cosmologies for robustness
    # (Omega_m, w0, w1)
    cosmo_params = [
        (0.3, -1.0, 0.0),
        (0.3, -1.2, 0.0),
        (0.3, -0.8, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -1.5, 0.3),
        (0.3, -0.5, -0.3),
        (0.35, -1.1, 0.2),
        (0.25, -1.1, 0.1),
        (0.3, -0.9, -0.2),
        (0.3, -1.3, 0.4),
        (0.28, -1.0, 0.0),
        (0.32, -1.0, 0.0),
    ]
    
    # Dense grid of redshifts
    z_test = np.linspace(0.05, 2.0, 20)
    
    def E_func(z, Om, w0, w1):
        OL = 1.0 - Om
        # Handle z=0 carefully, though z_test starts at 0.05
        return np.sqrt(Om * (1.0 + z)**3 + OL * (1.0 + z)**(3.0 * (1.0 + w0 + w1 * z)))
    
    def true_integral(z, Om, w0, w1):
        def integrand(u):
            return 1.0 / E_func(z * u, Om, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=100)
        return res

    # Precompute true integrals
    num_cosmo = len(cosmo_params)
    num_z = len(z_test)
    true_vals = np.zeros((num_cosmo, num_z))
    
    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[i, j] = true_integral(z, Om, w0, w1)
            
    if time.time() > deadline:
        # Fallback: Gauss-Legendre
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    # Objective function: Maximize the minimum of (1 - relative_error) 
    # or Minimize the maximum relative error.
    # We minimize max_rel_err.
    
    def objective(params):
        # params: 16 floats (8 nodes, 8 weights)
        # We need to handle constraints. SLSQP handles them.
        nodes = params[:8]
        weights = params[8:]
        
        # Calculate errors
        max_err = 0.0
        for i in range(num_cosmo):
            Om, w0, w1 = cosmo_params[i]
            for j in range(num_z):
                z = z_test[j]
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                integrand_vals = 1.0 / E_vals
                quad_val = np.dot(weights, integrand_vals)
                
                true_val = true_vals[i, j]
                if true_val > 0:
                    err = abs(quad_val - true_val) / true_val
                    if err > max_err:
                        max_err = err
        return max_err

    # Constraints for SLSQP
    # 1. Nodes strictly increasing in (0, 1)
    # 2. Weights positive
    # 3. Weights sum to 1
    
    # Initial guess: Chebyshev nodes mapped to [0,1]
    # Roots of T_8(x) are cos((2k-1)pi / 16) for k=1..8
    k = np.arange(1, 9)
    cheb_nodes_unit = np.cos((2 * k - 1) * np.pi / 16) # In [-1, 1]
    init_nodes = 0.5 * (cheb_nodes_unit + 1.0) # In [0, 1]
    
    # Initial weights: Gauss-Chebyshev weights are all 1/8? 
    # For standard Chebyshev quadrature (roots of T_n), weights are 1/n for all i if integrating f(x)dx? 
    # No, Gauss-Chebyshev integrates f(x)/sqrt(1-x^2). 
    # For simple polynomial interpolation at Chebyshev nodes, the weights are not uniform.
    # Let's just start with uniform weights 1/8 and let the optimizer adjust.
    init_weights = np.ones(8) / 8.0
    
    x0 = np.concatenate([init_nodes, init_weights])
    
    # Define constraints
    # Inequality constraints: 
    # n_0 > 0
    # n_{i} > n_{i-1} + eps
    # n_7 < 1
    # w_i > 0
    # Equality: sum(w) = 1
    
    constraints = []
    
    # Node constraints
    constraints.append({'type': 'ineq', 'fun': lambda x: x[0]}) # n0 > 0
    for i in range(1, 8):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[i-1] - 1e-6}) # n_i > n_{i-1} + eps
    constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - x[7]}) # n7 < 1
    
    # Weight constraints
    for i in range(8):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[8+i]}) # w_i > 0
        
    # Sum of weights
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0})
    
    # Bounds
    bounds = [(0, 1)] * 8 + [(1e-6, 1.0)] * 8
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        
        if result.success:
            best_params = result.x
        else:
            # If SLSQP fails, try to see if it improved anyway
            # Or fall back to initial
            best_params = result.x 
            
    except Exception:
        best_params = x0
    
    # Check time
    if time.time() > deadline:
        # Return best found or fallback
        nodes = best_params[:8]
        weights = best_params[8:]
        # Normalize weights
        weights = weights / np.sum(weights)
        # Ensure nodes are valid
        if np.all(np.diff(nodes) > 0) and np.all(nodes > 0) and np.all(nodes < 1) and np.all(weights > 0):
            return {"nodes": nodes, "weights": weights}
        else:
            # Fallback
            n = 8
            nodes, weights = np.polynomial.legendre.leggauss(n)
            nodes = 0.5 * (nodes + 1.0)
            weights = 0.5 * weights
            weights = weights / weights.sum()
            return {"nodes": nodes, "weights": weights}

    nodes = best_params[:8]
    weights = best_params[8:]
    
    # Final cleanup
    weights = weights / np.sum(weights)
    
    # Ensure strict increase and bounds
    # If SLSQP respected constraints, this should be fine.
    # But let's be safe.
    if not (np.all(np.diff(nodes) > 0) and np.all(nodes > 0) and np.all(nodes < 1) and np.all(weights > 0)):
        # Fallback
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}
        
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / weights.sum()
        
    return {"nodes": nodes, "weights": weights}
