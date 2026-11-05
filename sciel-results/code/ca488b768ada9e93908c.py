"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Design an 8-node positive quadrature rule using moment-matching with
    Chebyshev polynomials. The nodes and weights are tuned to exactly integrate
    Chebyshev polynomials up to degree 7, which provides optimal approximation
    properties for smooth functions with bounded derivatives.
    """
    n = 8
    
    # Initial guess: Gauss-Legendre nodes mapped to [0,1]
    x_gl = np.array([
        -0.960289856497536,
        -0.796666477413626,
        -0.525532409916329,
        -0.183434642500288,
         0.183434642500288,
         0.525532409916329,
         0.796666477413626,
         0.960289856497536
    ])
    w_gl = np.array([
        0.101228536290376,
        0.222381034453374,
        0.313706645877887,
        0.362683783378362,
        0.362683783378362,
        0.313706645877887,
        0.222381034453374,
        0.101228536290376
    ])
    
    x0_nodes = 0.5 * (x_gl + 1.0)
    x0_weights = 0.5 * w_gl
    
    # Define the objective function: minimize error in integrating Chebyshev polynomials
    def objective(params):
        # Unpack parameters: first n are nodes (in log-space for positivity), next n are weights (in log-space)
        log_nodes = params[:n]
        log_weights = params[n:]
        
        nodes = np.exp(log_nodes)
        weights = np.exp(log_weights)
        
        # Check constraints
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        # Normalize weights to sum to 1
        weights_norm = weights / np.sum(weights)
        
        # Compute Chebyshev polynomials T_k(x) for k = 0, 1, ..., 7
        # Map nodes from [0,1] to [-1,1] for Chebyshev
        x_cheb = 2 * nodes - 1
        
        # Integrate Chebyshev polynomials analytically on [-1,1]
        # ∫_{-1}^{1} T_k(x) dx = 0 for odd k, 2/(1-k^2) for even k > 0, 2 for k=0
        def cheb_int(k):
            if k == 0:
                return 2.0
            elif k % 2 == 1:
                return 0.0
            else:
                return 2.0 / (1 - k**2)
        
        # Map the integral to [0,1]: ∫_0^1 f(x) dx = 0.5 * ∫_{-1}^1 f(2t-1) dt
        # So we need: 0.5 * sum_i w_i * T_k(2*x_i - 1) ≈ 0.5 * cheb_int(k)
        # i.e., sum_i w_i * T_k(2*x_i - 1) ≈ cheb_int(k)
        
        error = 0.0
        for k in range(8):
            # Evaluate Chebyshev polynomial T_k at nodes
            T_k = np.cos(k * np.arccos(np.clip(x_cheb, -1, 1)))
            
            # Quadrature approximation
            quad_val = np.sum(weights_norm * T_k)
            
            # Exact integral (mapped to [0,1])
            exact_val = 0.5 * cheb_int(k)
            
            # Weighted error (give more weight to higher-degree polynomials)
            weight = 1.0 / (1 + k)
            error += weight * (quad_val - exact_val)**2
        
        # Add penalty for weights not summing to 1
        weight_sum_err = (np.sum(weights_norm) - 1.0)**2
        error += 100 * weight_sum_err
        
        return error
    
    # Initial parameters in log-space
    x0 = np.concatenate([np.log(x0_nodes), np.log(x0_weights)])
    
    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15})
    
    # Extract optimized nodes and weights
    if result.success:
        log_nodes_opt = result.x[:n]
        log_weights_opt = result.x[n:]
        
        nodes = np.exp(log_nodes_opt)
        weights = np.exp(log_weights_opt)
        
        # Sort nodes (should already be sorted, but ensure)
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weights = weights[sort_idx]
        
        # Normalize weights to sum exactly to 1
        weights = weights / np.sum(weights)
        
        # Ensure strict positivity and ordering
        if np.any(np.diff(nodes) <= 0) or np.any(nodes <= 0) or np.any(nodes >= 1) or np.any(weights <= 0):
            # Fallback to Gauss-Legendre
            nodes = x0_nodes
            weights = x0_weights
    else:
        # Fallback to Gauss-Legendre
        nodes = x0_nodes
        weights = x0_weights
    
    # Final validation
    if not (np.all(np.diff(nodes) > 0) and np.all(nodes > 0) and np.all(nodes < 1) and np.all(weights > 0)):
        nodes = x0_nodes
        weights = x0_weights
    
    # Ensure weights sum to 1 within 1e-12
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
