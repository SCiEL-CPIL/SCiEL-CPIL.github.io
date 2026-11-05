"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    
    def E_of_a(a, w0, wa):
        """E(a) = H(a)/H0 for CPL model in flat universe"""
        # E(a) = a^(-3(1+w0+wa)) * (1-a)^(3*wa)
        # But for a in (0,1], we need to handle a=0
        a_safe = np.maximum(a, 1e-15)
        term1 = a_safe ** (-3.0 * (1.0 + w0 + wa))
        term2 = (1.0 - a_safe) ** (3.0 * wa)
        return term1 * term2
    
    def integrand_z(z_prime, w0, wa):
        """The integrand in z-space: 1/E(z') where E is evaluated at a=1/(1+z')"""
        a = 1.0 / (1.0 + z_prime)
        return 1.0 / E_of_a(a, w0, wa)
    
    def quadrature_estimate(nodes, weights, z_max, w0, wa):
        """Compute the quadrature estimate of the integral ∫₀^z dz'/E(z')"""
        z_nodes = z_max * nodes
        a_nodes = 1.0 / (1.0 + z_nodes)
        E_vals = E_of_a(a_nodes, w0, wa)
        return np.sum(weights / E_vals)
    
    def reference_integral(z_max, w0, wa, n_points=10000):
        """High-precision reference using fine trapezoidal rule"""
        z_grid = np.linspace(0, z_max, n_points)
        a_grid = 1.0 / (1.0 + z_grid)
        E_grid = E_of_a(a_grid, w0, wa)
        f_grid = 1.0 / E_grid
        # Handle singularity at z=0 if needed (E(1)=1 so f(1)=1, fine)
        return np.trapz(f_grid, z_grid)
    
    # Define a set of test cosmologies (w0, wa) pairs
    # Cover a range of dark energy models
    test_models = [
        (0.0, 0.0),      # Lambda CDM
        (-0.5, 0.0),     # Quintessence
        (-1.0, 1.0),     # CPL standard
        (-1.0, -1.0),    # Phantom
        (-0.5, 0.5),     
        (-1.5, 0.0),     # Phantom
        (-1.0, 0.5),     
        (-0.8, -0.5),    
        (-1.2, 0.3),     
        (-0.7, 0.2),     
    ]
    
    # Test redshifts
    test_zs = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    
    def compute_errors(nodes, weights):
        """Compute relative errors for all test cases"""
        errors = []
        for w0, wa in test_models:
            for z in test_zs:
                quad_val = quadrature_estimate(nodes, weights, z, w0, wa)
                ref_val = reference_integral(z, w0, wa)
                if abs(ref_val) > 1e-10:
                    rel_err = abs(quad_val - ref_val) / abs(ref_val)
                    errors.append(rel_err)
        return np.array(errors)
    
    def objective(x):
        """Minimize worst-case relative error (smoothed)"""
        nodes_raw = x[:8]
        weights_raw = x[8:16]
        
        # Enforce strictly increasing nodes in (0,1) via softplus-like transform
        # Use cumulative sum of positive values
        # nodes[0] = sigmoid(x[0]), nodes[i] = nodes[i-1] + sigmoid(x[i]) * (1 - nodes[i-1])
        # Simpler: use tanh to map to (0,1) and enforce ordering
        # Let's use: nodes_i = (tanh(x_i) + 1)/2, then sort and ensure strict increase
        # Better: parameterize as node_i = node_{i-1} + exp(y_i) * (1 - node_{i-1})
        
        # Use a different parameterization:
        # Let d_i > 0 be gaps, nodes = cumsum of normalized gaps
        # d_i = exp(x[i-8]) for i=0..7, but we need 8 gaps... 
        # Actually 8 nodes means 8 positions. Let's use:
        # node_i = (i + 1 + u_i) / (n+1) where u_i is small perturbation
        # But that constrains too much.
        
        # Alternative: nodes_i = sigmoid(x_i), then sort
        # But sorting breaks differentiability.
        
        # Let's use: nodes are parameterized by 8 values in (0,1) via sigmoid,
        # and we add a penalty if they're not strictly increasing.
        
        nodes = 1.0 / (1.0 + np.exp(-nodes_raw))
        # Penalize non-increasing
        diffs = np.diff(nodes)
        order_penalty = np.sum(np.maximum(0, -diffs + 1e-6)) ** 2
        
        # Weights: positive via softplus
        weights = np.logaddexp(0, weights_raw)  # softplus
        w_sum = np.sum(weights)
        weights = weights / w_sum  # normalize
        
        if w_sum < 1e-10:
            return 1e6
        
        errors = compute_errors(nodes, weights)
        worst_err = np.max(errors)
        
        # Smooth max via log-sum-exp
        beta = 50.0
        smooth_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * errors)))
        
        return smooth_max + order_penalty + 1e-10
    
    # Initial guess: midpoint rule
    n = 8
    x0_nodes = np.zeros(8)  # sigmoid(0) = 0.5, all same... bad
    # Better initial: evenly spaced
    initial_nodes = (np.arange(n) + 0.5) / n
    # Inverse sigmoid
    x0_nodes = np.log(initial_nodes / (1 - initial_nodes))
    x0_weights = np.zeros(8)  # softplus(0) = ln(2), all equal
    
    x0 = np.concatenate([x0_nodes, x0_weights])
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 70:
        # Return midpoint rule
        return {"nodes": initial_nodes, "weights": np.full(n, 1.0/n)}
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    elapsed = time.time() - start_time
    if elapsed > 80:
        # Return best found so far or midpoint
        if result.success or result.fun < 1e-3:
            nodes_raw = result.x[:8]
            weights_raw = result.x[8:16]
            nodes = 1.0 / (1.0 + np.exp(-nodes_raw))
            weights = np.logaddexp(0, weights_raw)
            weights = weights / np.sum(weights)
            # Ensure strictly increasing
            for i in range(1, 8):
                if nodes[i] <= nodes[i-1]:
                    nodes[i] = nodes[i-1] + 1e-6
            nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
            return {"nodes": nodes, "weights": weights}
        else:
            return {"nodes": initial_nodes, "weights": np.full(n, 1.0/n)}
    
    # Extract solution
    nodes_raw = result.x[:8]
    weights_raw = result.x[8:16]
    nodes = 1.0 / (1.0 + np.exp(-nodes_raw))
    weights = np.logaddexp(0, weights_raw)
    weights = weights / np.sum(weights)
    
    # Ensure strictly increasing nodes in (0,1)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1] + 1e-8:
            nodes[i] = nodes[i-1] + 1e-6
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    
    # Final polish: do a few more iterations if time permits
    elapsed = time.time() - start_time
    if elapsed < 85:
        x_polish = result.x
        result2 = minimize(objective, x_polish, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
        nodes_raw = result2.x[:8]
        weights_raw = result2.x[8:16]
        nodes = 1.0 / (1.0 + np.exp(-nodes_raw))
        weights = np.logaddexp(0, weights_raw)
        weights = weights / np.sum(weights)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1] + 1e-8:
                nodes[i] = nodes[i-1] + 1e-6
        nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
