"""Model-and-optimize approach: directly optimize nodes and weights for luminosity distance quadrature."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 88.0

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

# Precompute high-accuracy Gauss-Legendre nodes/weights for reference integration
# 64-point GL is accurate to machine precision for smooth functions like 1/E(a)
_REF_NODES, _REF_WEIGHTS = np.polynomial.legendre.leggauss(64)
# Map from [-1, 1] to [0, 1]
_REF_NODES = 0.5 * (_REF_NODES + 1.0)
_REF_WEIGHTS = 0.5 * _REF_WEIGHTS

def _true_dL(z, w0, wa):
    # Integrate 1/E(a) from z=0 to z=z
    # Change of variable: a = 1/(1+z'). z' goes 0 to z.
    # a goes 1 to 1/(1+z).
    # Integral I = \int_0^z 1/E(1/(1+z')) dz'
    # Using GL on [0, z]:
    if z == 0:
        return 0.0
    # Map reference nodes [0,1] to [0, z]
    z_nodes = _REF_NODES * z
    a_vals = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(_REF_WEIGHTS / E_vals) * z
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _worst_rel_error(nodes, weights, test_cosmos, test_zs):
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
    return max_err

def _softmax_error(nodes, weights, test_cosmos, test_zs, beta=100.0):
    """Smooth approximation to max error using log-sum-exp."""
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
    # Use log-sum-exp for numerical stability
    max_e = np.max(errors)
    if max_e < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (errors - max_e)))) / beta + max_e
    return log_sum

def _gauss_legendre_init():
    """Get 8-point Gauss-Legendre nodes and weights on [0,1]."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    weights = weights / np.sum(weights)
    return nodes, weights

def _decode_params(params):
    """
    Decode parameters to nodes and weights.
    Params 0-7: node positions via tanh to ensure (0,1)
    Params 8-15: log-weights, normalized
    """
    # Nodes: use a parameterization that ensures (0,1) and monotonicity
    # t_i = params[i], nodes_i = (i + 1 + exp(t_i)) / (n + sum(exp(t)))
    # Actually simpler: use sorted sigmoid transforms
    raw = params[:8]
    # Map to (0,1) via sigmoid, then enforce ordering
    sig = 1.0 / (1.0 + np.exp(-10.0 * (raw - 3.5)))
    # sig is in (0,1), now sort and ensure strictly increasing
    sig = np.sort(sig)
    # Add small spacing to ensure strict increase
    for i in range(1, 8):
        if sig[i] <= sig[i-1] + 1e-12:
            sig[i] = sig[i-1] + 1e-12
    # Clip away from boundaries
    sig = np.clip(sig, 1e-10, 1.0 - 1e-10)
    nodes = sig
    
    # Weights: exponential of params, normalized
    log_w = params[8:16]
    w_raw = np.exp(log_w)
    weights = w_raw / np.sum(w_raw)
    
    return nodes, weights

def _encode_initial(nodes, weights):
    """Encode initial nodes/weights to parameter space."""
    # For nodes: invert sigmoid
    params_nodes = np.log(nodes / (1.0 - nodes)) / 10.0 + 3.5
    # For weights: log
    params_w = np.log(weights)
    return np.concatenate([params_nodes, params_w])

def _objective(params):
    if _time_left() < 3.0:
        return 1e10
    
    nodes, weights = _decode_params(params)
    
    # Representative test set
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   # Another CPL
        (0.3, -0.3),   # Mild CPL
        (-2.0, 1.0),   # More phantom
        (1.0, -2.0),   # More quintessence
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    # Use softmax for smoother optimization
    return _softmax_error(nodes, weights, test_cosmos, test_zs, beta=80.0)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 88.0
    
    # Start from Gauss-Legendre
    gl_nodes, gl_weights = _gauss_legendre_init()
    params_init = _encode_initial(gl_nodes, gl_weights)
    
    best_params = params_init.copy()
    best_score = _objective(best_params)
    
    # Multi-start with different initializations
    # We can afford many more restarts now because _true_dL is fast
    np.random.seed(42)
    initializations = [params_init.copy()]
    
    # Generate many random perturbations
    for _ in range(30):
        perturbation = np.random.randn(16) * 0.5
        initializations.append(params_init + perturbation)
        
    # Also try some structured perturbations
    for _ in range(10):
        # Perturb nodes more strongly
        p = params_init.copy()
        p[:8] += np.random.randn(8) * 1.0
        initializations.append(p)
        
    # Try each initialization
    for init_params in initializations:
        if _time_left() < 5.0:
            break
        
        try:
            # Nelder-Mead for non-smooth objective
            res = minimize(_objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'initial_simplex': 0.1})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final refinement with tighter tolerance
    if _time_left() > 5.0:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Decode final parameters
    nodes, weights = _decode_params(best_params)
    
    # Ensure validity: nodes in (0,1) strictly increasing, weights positive and sum to 1
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    weights = np.abs(weights)
    if np.sum(weights) < 1e-15:
        weights = np.ones(8) / 8.0
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
