"""Gauss-Legendre-based 8-node quadrature for luminosity distance with targeted refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL(z, w0, wa):
    def integrand(z_prime):
        a = 1.0 / (1.0 + z_prime)
        return 1.0 / _E_cpl(a, w0, wa)
    val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * val

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _worst_rel_error(nodes, weights, test_cosmos, test_zs, beta=50.0):
    """
    Compute worst relative error with optional soft-min smoothing.
    beta controls smoothness: beta->inf gives true max, beta=0 gives mean.
    """
    errors = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                errors.append(rel_err)
    
    if not errors:
        return 0.0
    
    errors = np.array(errors)
    
    # Soft-min: -log(sum(exp(-beta * errors))) / beta
    # As beta -> inf, this approaches max(errors)
    # For numerical stability, shift by max
    max_e = np.max(errors)
    if beta < 1.0:
        return float(max_e)  # Use true max for low beta
    shifted = -beta * (errors - max_e)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 0)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min = max_e - log_sum_exp / beta
    
    return float(soft_min)

def _gauss_legendre_init():
    """Get 8-point Gauss-Legendre nodes and weights on [0,1]."""
    # Standard Gauss-Legendre on [-1,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    # Map to [0,1]: x_new = 0.5*(x + 1), w_new = 0.5*w
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    # Sort nodes to be strictly increasing
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    return nodes, weights

def _params_to_nodes_weights(params):
    """
    Direct parameterization: params[0:8] are logit-transformed nodes,
    params[8:15] are log-weights (last weight determined by sum=1 constraint).
    """
    # Nodes: use logit transform to ensure (0,1)
    # node_i = sigmoid(params[i]) = 1/(1+exp(-params[i]))
    # This gives a smooth mapping from R to (0,1)
    raw_nodes = 1.0 / (1.0 + np.exp(-params[:8]))
    
    # Ensure strictly increasing by sorting and adding small offsets if needed
    # First, sort
    sort_idx = np.argsort(raw_nodes)
    sorted_nodes = raw_nodes[sort_idx]
    
    # If any nodes are too close, push them apart
    for i in range(1, 8):
        if sorted_nodes[i] - sorted_nodes[i-1] < 1e-10:
            sorted_nodes[i] = sorted_nodes[i-1] + 1e-10
    
    # Clip to valid range
    sorted_nodes = np.clip(sorted_nodes, 1e-8, 1.0 - 1e-8)
    
    # Weights: use softmax on first 7 weights, last weight = 1 - sum(first 7)
    # This ensures positive weights that sum to 1
    log_w = params[8:15]
    # Softmax
    max_log_w = np.max(log_w)
    exp_w = np.exp(log_w - max_log_w)
    w_sum = np.sum(exp_w)
    weights = np.zeros(8)
    weights[:7] = exp_w / w_sum
    weights[7] = 1.0 - np.sum(weights[:7])
    
    # If last weight is not positive, fall back to uniform
    if weights[7] <= 0:
        weights = np.ones(8) / 8.0
    else:
        # Ensure all weights are positive
        weights = np.maximum(weights, 1e-15)
        weights = weights / np.sum(weights)
    
    return sorted_nodes, weights

def _objective(params, beta=50.0):
    if _time_left() < 2.0:
        return 1e10
    
    nodes, weights = _params_to_nodes_weights(params)
    
    # Test set: representative cosmologies and redshifts
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   # Another CPL
        (0.3, -0.3),   # Mild CPL
        (1.0, -1.5),   # Strong quintessence
        (-0.8, 0.2),   # Near-CDM
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    return _worst_rel_error(nodes, weights, test_cosmos, test_zs, beta=beta)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Start from Gauss-Legendre
    gl_nodes, gl_weights = _gauss_legendre_init()
    
    # Initial params: logit of GL nodes, log of GL weights (first 7)
    params_init = np.zeros(16)
    # Logit transform: x = log(node/(1-node))
    for i in range(8):
        if gl_nodes[i] > 0 and gl_nodes[i] < 1:
            params_init[i] = np.log(gl_nodes[i] / (1.0 - gl_nodes[i]))
    # Log weights for first 7
    for i in range(7):
        params_init[8+i] = np.log(gl_weights[i])
    # 8th weight is determined by constraint
    
    best_params = params_init.copy()
    best_score = _objective(best_params, beta=50.0)
    
    # Phase 1: Coarse optimization with moderate soft-min smoothing
    if _time_left() > 30.0:
        try:
            res = minimize(lambda p: _objective(p, beta=50.0), params_init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Phase 2: Fine-tune with stronger soft-min (closer to true max)
    if _time_left() > 20.0:
        try:
            res = minimize(lambda p: _objective(p, beta=200.0), best_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Phase 3: High-precision polish with true max objective
    if _time_left() > 10.0:
        try:
            res = minimize(lambda p: _objective(p, beta=1000.0), best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Phase 4: Final local refinement with true max
    if _time_left() > 5.0:
        try:
            res = minimize(lambda p: _objective(p, beta=10000.0), best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final decode
    nodes, weights = _params_to_nodes_weights(best_params)
    
    # Ensure validity
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
