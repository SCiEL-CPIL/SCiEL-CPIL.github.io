"""Direct 8-node quadrature construction using fast reference for CPL cosmologies."""
import time
import numpy as np
from scipy.optimize import minimize

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

def _chebyshev_nodes_weights(n=8):
    """Gauss-Legendre nodes and weights on (0,1)."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    nodes_gl = 0.5 * (x_gl + 1.0)
    weights_gl = 0.5 * w_gl
    idx = np.argsort(nodes_gl)
    nodes_gl = nodes_gl[idx]
    weights_gl = weights_gl[idx]
    weights_gl = weights_gl / np.sum(weights_gl)
    return nodes_gl, weights_gl

def _params_to_nodes_weights(params):
    """
    Direct parameterization: params 0-7 are node positions via sigmoid,
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

def _fast_objective(params):
    """
    Fast objective using 64-point Gauss-Legendre as reference (no quad).
    """
    if _time_left() < 5.0:
        return 1e10
    nodes, weights = _params_to_nodes_weights(params)
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(64)
    ref_nodes = 0.5 * (ref_nodes + 1.0)
    ref_weights = 0.5 * ref_weights
    test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (1.0, -2.0), (-2.0, 1.0),
        (0.0, 1.0), (-1.0, 1.0), (0.0, -1.0), (-0.5, -0.5),
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            a_ref = 1.0 / (1.0 + z * ref_nodes)
            E_ref = _E_cpl(a_ref, w0, wa)
            ref_integral = np.sum(ref_weights / E_ref)
            true_dL = (1.0 + z) * z * ref_integral
            a_8 = 1.0 / (1.0 + z * nodes)
            E_8 = _E_cpl(a_8, w0, wa)
            approx_integral = np.sum(weights / E_8)
            approx_dL = (1.0 + z) * z * approx_integral
            if true_dL > 1e-10:
                rel_err = abs(approx_dL - true_dL) / true_dL
                if rel_err > max_err:
                    max_err = rel_err
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 82.0
    gl_nodes, gl_weights = _chebyshev_nodes_weights(8)
    eps = 1e-10
    gl_nodes_safe = np.clip(gl_nodes, eps, 1-eps)
    params_init = np.zeros(16)
    params_init[:8] = np.log(gl_nodes_safe / (1 - gl_nodes_safe))
    params_init[8:16] = np.log(gl_weights)
    best_params = params_init.copy()
    best_score = _fast_objective(best_params)
    if _time_left() > 20.0:
        try:
            res = minimize(_fast_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    if _time_left() > 15.0:
        try:
            res = minimize(_fast_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    nodes, weights = _params_to_nodes_weights(best_params)
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = weights / np.sum(weights)
    return {"nodes": nodes, "weights": weights}
