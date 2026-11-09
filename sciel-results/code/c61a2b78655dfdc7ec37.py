"""Direct 8-node quadrature construction using global search and pre-computed references."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 82.0

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

def _precompute_references():
    """Pre-compute reference integrals for a set of cosmologies and redshifts."""
    test_cosmos = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.5, -1.0),
        (-1.5, 0.5),
        (-0.5, 0.5),
        (0.3, -0.3),
        (1.0, -2.0),
        (-2.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (0.0, -1.0),
        (-0.5, -0.5),
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    # For each (w0, wa, z), pre-compute the reference integral using dense quadrature
    # The integral is: ∫_0^1 1/E(1/(1+z*t)) dt
    # We'll use Gauss-Legendre with many points as reference
    n_ref = 200
    x_ref, w_ref = np.polynomial.legendre.leggauss(n_ref)
    t_ref = 0.5 * (x_ref + 1.0)  # map to [0,1]
    w_ref = 0.5 * w_ref
    
    ref_integrals = np.zeros((len(test_cosmos), len(test_zs)))
    for ci, (w0, wa) in enumerate(test_cosmos):
        for zi, z in enumerate(test_zs):
            a_vals = 1.0 / (1.0 + z * t_ref)
            E_vals = _E_cpl(a_vals, w0, wa)
            ref_integrals[ci, zi] = np.sum(w_ref / E_vals)
    
    return test_cosmos, test_zs, ref_integrals

def _params_to_nodes_weights(params):
    """
    Parameterization: params 0-7 are node positions via sigmoid,
    params 8-15 are log-weights.
    """
    raw_nodes = 1.0 / (1.0 + np.exp(-params[:8]))
    raw_nodes = np.sort(raw_nodes)
    for i in range(1, 8):
        if raw_nodes[i] <= raw_nodes[i-1]:
            raw_nodes[i] = raw_nodes[i-1] + 1e-10
    raw_nodes = np.clip(raw_nodes, 1e-10, 1.0 - 1e-10)
    
    log_weights = params[8:16]
    w_raw = np.exp(log_weights)
    weights = w_raw / np.sum(w_raw)
    
    return raw_nodes, weights

def _fast_objective(params, test_cosmos, test_zs, ref_integrals):
    """Fast objective using pre-computed references."""
    if _time_left() < 2.0:
        return 1e10
    
    nodes, weights = _params_to_nodes_weights(params)
    
    # Compute quadrature approximation for all test cases
    # dL_quad = (1+z) * z * sum_i weights[i] / E(a_i)
    # dL_true = (1+z) * ref_integral
    # Relative error = |quad - true| / true
    
    max_err = 0.0
    for ci, (w0, wa) in enumerate(test_cosmos):
        for zi, z in enumerate(test_zs):
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa)
            integral_approx = np.sum(weights / E_vals)
            
            quad_val = (1.0 + z) * z * integral_approx
            true_val = (1.0 + z) * ref_integrals[ci, zi]
            
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
    
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 82.0
    
    # Pre-compute references
    test_cosmos, test_zs, ref_integrals = _precompute_references()
    
    # Initial guess: Gauss-Legendre
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    gl_nodes = 0.5 * (x_gl + 1.0)
    gl_weights = 0.5 * w_gl
    idx = np.argsort(gl_nodes)
    gl_nodes = gl_nodes[idx]
    gl_weights = gl_weights[idx]
    gl_weights = gl_weights / np.sum(gl_weights)
    
    eps = 1e-10
    gl_nodes_safe = np.clip(gl_nodes, eps, 1-eps)
    params_init = np.zeros(16)
    params_init[:8] = np.log(gl_nodes_safe / (1 - gl_nodes_safe))
    params_init[8:16] = np.log(gl_weights)
    
    best_params = params_init.copy()
    best_score = _fast_objective(best_params, test_cosmos, test_zs, ref_integrals)
    
    # Differential Evolution for global search
    if _time_left() > 25.0:
        try:
            bounds = [(-5.0, 5.0)] * 16  # sigmoid bounds for nodes, log weight bounds
            
            def de_objective(p):
                return _fast_objective(p, test_cosmos, test_zs, ref_integrals)
            
            res = differential_evolution(
                de_objective, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-12, mutation=(0.3, 1.0), recombination=0.8,
                polish=False, init='sobol',
                update_interval=1, callback=None
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Nelder-Mead refinement
    if _time_left() > 15.0:
        try:
            res = minimize(
                lambda p: _fast_objective(p, test_cosmos, test_zs, ref_integrals),
                best_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final refinement
    if _time_left() > 10.0:
        try:
            res = minimize(
                lambda p: _fast_objective(p, test_cosmos, test_zs, ref_integrals),
                best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
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
