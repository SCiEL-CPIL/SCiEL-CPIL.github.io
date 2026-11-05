import numpy as np
from scipy.optimize import minimize
import time

def construct():
    t0 = time.time()
    deadline = 80.0  # safety margin before 90s kill

    # Define a diverse set of CPL (Chevallier-Polarski-Linder) models
    # w0, wa pairs covering a broad range
    models = []
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for w0 in w0_vals:
        for wa in wa_vals:
            models.append((w0, wa))
    # Add some extreme cases
    models.append((-2.0, 2.0))
    models.append((1.0, -1.0))
    models.append((-1.0, -1.0))
    models.append((-1.0, 1.0))

    def E_of_a(a, w0, wa):
        """E(a) = H(a)/H0 for flat CPL model"""
        # E^2 = (1/a)^3 + Omega_m*(1/a)^3 + Omega_de*(1/a)^(3(1+w0+wa*(1-1/a)))
        # More precisely: E^2 = (1+z)^3 + Omega_de * (1+z)^(3(1+w))
        # where 1+w = 1+w0+wa*(1-a) = 1+w0+wa*z
        # Actually: E^2(a) = a^{-3} + Omega_de * a^{-3(1+w0+wa*(1-a))}
        # Let me use the standard form: E^2 = (1+z)^3 + Omega_de*(1+z)^{3(1+w0+wa*z)}
        # with a = 1/(1+z), so 1+z = 1/a
        # E^2 = a^{-3} + Omega_de * a^{-3(1+w0+wa*(1-a))}
        # Simplify: 3(1+w0+wa*(1-a)) = 3+3w0+3wa-3wa*a
        Om_m = 0.3
        Om_de = 1.0 - Om_m
        z = 1.0 / a - 1.0
        w = w0 + wa * z
        E2 = (1.0 + z)**3 + Om_de * (1.0 + z)**(3.0 * (1.0 + w))
        return np.sqrt(np.maximum(E2, 1e-30))

    def integrand(z, w0, wa):
        a = 1.0 / (1.0 + z)
        return (1.0 + z) * z / E_of_a(a, w0, wa)

    def reference_integral(w0, wa, n_points=10000):
        """High-accuracy reference using Simpson's rule on fine grid"""
        z = np.linspace(0, 1, n_points)
        f = np.array([integrand(zi, w0, wa) for zi in z])
        # Simpson's rule
        h = 1.0 / (n_points - 1)
        # Ensure even number of intervals
        if (n_points - 1) % 2 != 0:
            n_points -= 1
            z = np.linspace(0, 1, n_points)
            f = np.array([integrand(zi, w0, wa) for zi in z])
            h = 1.0 / (n_points - 1)
        result = h/3.0 * (f[0] + f[-1] + 4*np.sum(f[1:-1:2]) + 2*np.sum(f[2:-1:2]))
        return result

    # Precompute reference integrals
    refs = []
    for (w0, wa) in models:
        refs.append(reference_integral(w0, wa))
    refs = np.array(refs)

    def quadrature_value(nodes, weights, w0, wa):
        a = 1.0 / (1.0 + nodes)
        E_vals = E_of_a(a, w0, wa)
        return np.sum(weights * (1.0 + nodes) * nodes / E_vals)

    def objective(params):
        # params: 8 node parameters (unconstrained), 8 weight parameters (unconstrained)
        # Transform to ensure nodes in (0,1) strictly increasing and weights > 0 summing to 1
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Nodes: use softplus-like transformation for strict ordering
        # Start with base grid, add small perturbations
        base_nodes = np.linspace(0.1, 0.9, 8)
        nodes = base_nodes + 0.15 * np.tanh(node_params)
        # Ensure strict ordering by construction: use cumulative positive increments
        # Better: use a parameterization that guarantees ordering
        # Let's use: node_i = sigmoid(cumsum of transformed params)
        # Simpler: just clamp and ensure ordering
        nodes = np.clip(nodes, 0.01, 0.99)
        # Enforce strict ordering
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 0.001
        nodes = np.clip(nodes, 0.001, 0.999)
        
        # Weights: softmax to ensure positive, then normalize
        w_raw = np.exp(weight_params - np.max(weight_params))
        weights = w_raw / np.sum(w_raw)
        
        # Compute max relative error across all models
        max_err = 0.0
        for i, (w0, wa) in enumerate(models):
            q_val = quadrature_value(nodes, weights, w0, wa)
            ref_val = refs[i]
            if ref_val > 1e-10:
                rel_err = abs(q_val - ref_val) / abs(ref_val)
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    # Initial guess: midpoint rule
    x0 = np.zeros(16)
    # Set initial nodes to midpoint positions
    mid_nodes = (np.arange(8) + 0.5) / 8.0
    base_nodes = np.linspace(0.1, 0.9, 8)
    # Invert the transformation approximately
    x0[:8] = np.arctanh((mid_nodes - base_nodes) / 0.15)
    x0[8:16] = 0.0  # equal weights

    best_params = x0.copy()
    best_err = objective(x0)
    
    # Try multiple optimization strategies
    strategies = [
        {'method': 'Nelder-Mead', 'options': {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}},
        {'method': 'Powell', 'options': {'maxiter': 500, 'ftol': 1e-10}},
    ]
    
    # Also try different initial points
    initial_points = [x0]
    # Random perturbations
    np.random.seed(42)
    for _ in range(5):
        pert = x0 + 0.5 * np.random.randn(16)
        initial_points.append(pert)
    
    for strat in strategies:
        if time.time() - t0 > deadline - 10:
            break
        for ip in initial_points:
            if time.time() - t0 > deadline - 10:
                break
            try:
                res = minimize(objective, ip, method=strat['method'], options=strat['options'])
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Extract best nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    base_nodes = np.linspace(0.1, 0.9, 8)
    nodes = base_nodes + 0.15 * np.tanh(node_params)
    nodes = np.clip(nodes, 0.01, 0.99)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 0.001
    nodes = np.clip(nodes, 0.001, 0.999)
    
    w_raw = np.exp(weight_params - np.max(weight_params))
    weights = w_raw / np.sum(w_raw)
    
    # Final validation: ensure sum to 1 within 1e-12
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
