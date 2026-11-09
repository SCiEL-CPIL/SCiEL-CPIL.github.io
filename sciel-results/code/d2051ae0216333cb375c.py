"""Rewritten program for luminosity-distance quadrature optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


def _E_cpl(a, w0, wa):
    """CPL dark energy model for flat universe."""
    Om_m = 0.3
    Om_r = 5e-5
    Om_DE = 1.0 - Om_m - Om_r
    w = w0 + wa * (1 - a)
    return np.sqrt(Om_m * a**(-3) + Om_r * a**(-4) + Om_DE * a**(-3*(1+w0+wa)) * (1 + wa*(1-a)))


def _integrand(z, w0, wa):
    """The integrand for luminosity distance: z / E(a) where a = 1/(1+z)"""
    a = 1.0 / (1.0 + z)
    return z / _E_cpl(a, w0, wa)


def _true_dL(z, w0, wa):
    """Compute true dimensionless luminosity distance using adaptive quadrature."""
    val, _ = quad(_integrand, 0, z, args=(w0, wa), limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1 + z) * val


def _quadrature_dL(z, nodes, weights, w0, wa):
    """Compute quadrature approximation of dimensionless luminosity distance."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(weights / E_vals)
    return z * integral


def _worst_rel_error(nodes, weights, test_zs, test_cosmos):
    """Compute worst relative error (in percent) over test set."""
    max_err = 0.0
    for (w0, wa) in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quadrature_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val * 100.0
                if rel_err > max_err:
                    max_err = rel_err
    return max_err


def _params_to_nodes_weights(params):
    """Convert unconstrained parameters to valid nodes and weights.
    
    nodes: 7 free parameters -> 8 strictly increasing nodes in (0,1)
    weights: 7 free parameters -> 8 positive weights summing to 1
    
    Use log-cumsum for nodes to ensure strict monotonicity.
    Use softmax for weights.
    """
    n = 8
    node_params = params[:7]
    weight_params = params[7:14]
    
    # Nodes: start at a small value, add positive increments
    # node_0 = sigmoid(node_params[0]) * 0.05 + 0.001
    # node_i = node_{i-1} + sigmoid(node_params[i-1]) * (1 - node_{i-1}) * 0.3
    
    # Better: use cumulative sum of exponentials
    base = 0.01
    increments = np.exp(node_params) * 0.15
    nodes = np.cumsum(increments)
    nodes = base + nodes / nodes[-1] * (1 - 2*base)
    
    # Ensure strictly in (0, 1)
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    
    # Weights: softmax
    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)
    
    return nodes, weights


def _objective(params, test_zs, test_cosmos, beta=50.0):
    """Soft-max surrogate of worst-case relative error."""
    nodes, weights = _params_to_nodes_weights(params)
    
    # Check validity
    if not np.all(np.diff(nodes) > 1e-10):
        return 1e10
    if not np.all(weights > 1e-15):
        return 1e10
    if abs(np.sum(weights) - 1.0) > 1e-8:
        return 1e10
    
    errors = []
    for (w0, wa) in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quadrature_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val * 100.0
                errors.append(rel_err)
    
    if not errors:
        return 1e10
    
    errors = np.array(errors)
    # Soft max: approximates max with smooth gradient
    max_e = np.max(errors)
    soft_max = max_e + np.log(np.sum(np.exp((errors - max_e) / beta)) * beta)
    return soft_max


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 8
    
    # Initialize with Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    gl_nodes = 0.5 * (gl_nodes + 1)
    gl_weights = 0.5 * gl_weights
    
    # Invert the parameterization to get initial params
    # This is tricky, so let's just start from a reasonable random init
    # and let the optimizer find good values
    
    # Test set for optimization
    test_zs = np.array([0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    test_cosmos = [
        (0.0, 0.0),      # LCDM
        (-1.0, 0.0),     # Cosmological constant
        (0.5, -1.0),     # Quintessence-like
        (-1.5, 0.5),     # Phantom-like
        (-0.5, 1.0),     # Another model
        (1.0, -2.0),     # Exotic
    ]
    
    # Initialize params: we want nodes close to Gauss-Legendre
    # Use a simple random init centered around 0
    np.random.seed(42)
    x0 = np.zeros(14)
    
    # Better initialization: try to match Gauss-Legendre
    # The parameterization is complex, so let's just use random init
    # and rely on the optimizer
    
    # Run Nelder-Mead optimization
    best_result = None
    best_score = 1e10
    
    try:
        result = minimize(
            _objective,
            x0,
            args=(test_zs, test_cosmos, 50.0),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
        
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    if best_result is not None and remaining > 5:
        # Try L-BFGS-B as a second optimizer with the best Nelder-Mead result
        try:
            result2 = minimize(
                _objective,
                best_result,
                args=(test_zs, test_cosmos, 50.0),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_result = result2.x
                
        except Exception:
            pass
    
    if best_result is not None:
        nodes, weights = _params_to_nodes_weights(best_result)
    else:
        # Fallback to Gauss-Legendre
        nodes = gl_nodes
        weights = gl_weights
    
    # Final validation and cleanup
    nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
    
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
