"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Design custom quadrature nodes and weights optimized for
    # dimensionless luminosity distance integrals in flat CPL dark energy models.
    # 
    # The integrand is f(z) = z / E(a) where a = 1/(1+z) and
    # E(a) = (a/0)^(-3(1+w0+wa)) * (a/0)^(-3wa) for CPL model.
    # 
    # We optimize nodes and weights to minimize worst-case relative error
    # across a family of CPL parameters (w0, wa) that cover the physically
    # relevant range.
    
    # Define a set of test CPL models covering physically relevant parameter space
    # w0 in [-1.5, 0], wa in [-1.0, 1.0]
    w0_values = np.linspace(-1.5, 0.0, 7)
    wa_values = np.linspace(-1.0, 1.0, 7)
    
    def E_of_a(a, w0, wa):
        """CPL dark energy: E(a) = (a/0)^(-3(1+w0+wa)) * (a/0)^(-3wa)"""
        # For flat universe with matter + CPL dark energy:
        # E^2(a) = a^(-3) + (1 - a^(-3)) * a^(-3(1+w0+wa))
        # Simplified: E(a) = sqrt(a^(-3) + (1 - a^(-3)) * a^(-3(1+w0+wa)))
        a_safe = np.maximum(a, 1e-10)
        matter = a_safe**(-3)
        de = (1 - a_safe**(-3)) * a_safe**(-3*(1+w0+wa))
        return np.sqrt(np.maximum(matter + de, 1e-20))
    
    def integrand(z, w0, wa):
        """f(z) = z / E(a) where a = 1/(1+z)"""
        a = 1.0 / (1.0 + z)
        return z / E_of_a(a, w0, wa)
    
    def adaptive_integral(w0, wa, n_points=10000):
        """Compute reference integral using fine trapezoidal rule"""
        z = np.linspace(0, 1, n_points)
        f = integrand(z, w0, wa)
        return np.trapz(f, z)
    
    def quadrature_value(nodes, weights, w0, wa):
        """Compute quadrature approximation"""
        z = nodes
        f = integrand(z, w0, wa)
        return np.dot(weights, f)
    
    def objective(params):
        """Minimize worst-case relative error across test models"""
        # params: 8 nodes + 8 weights = 16 parameters
        nodes = params[:8]
        weights = params[8:]
        
        # Enforce constraints softly in objective
        # Nodes must be in [0,1] and strictly increasing
        node_diffs = np.diff(nodes)
        if np.any(node_diffs <= 0):
            return 1e10
        if np.any(nodes < 0) or np.any(nodes > 1):
            return 1e10
        
        # Weights must be positive
        if np.any(weights <= 0):
            return 1e10
        
        # Sum of weights should be 1
        weight_sum_error = abs(np.sum(weights) - 1.0)
        if weight_sum_error > 1e-6:
            return 1e10 + weight_sum_error
        
        # Compute worst-case relative error
        max_rel_err = 0.0
        for w0 in w0_values:
            for wa in wa_values:
                ref = adaptive_integral(w0, wa)
                approx = quadrature_value(nodes, weights, w0, wa)
                if ref > 1e-10:
                    rel_err = abs(approx - ref) / ref
                    max_rel_err = max(max_rel_err, rel_err)
        
        return max_rel_err
    
    # Initial guess: Gauss-Legendre nodes and weights
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = (x_gl + 1.0) / 2.0
    weights_init = w_gl / 2.0
    
    params_init = np.concatenate([nodes_init, weights_init])
    
    # Optimize using Nelder-Mead (no gradient needed)
    result = minimize(objective, params_init, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                               'adaptive': True})
    
    if result.success:
        nodes = result.x[:8]
        weights = result.x[8:]
    else:
        # Fallback to Gauss-Legendre if optimization fails
        nodes = nodes_init
        weights = weights_init
    
    # Final adjustments to ensure constraints are met
    # Ensure nodes are strictly in (0, 1)
    nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    # Ensure weights are positive
    weights = np.maximum(weights, 1e-10)
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
