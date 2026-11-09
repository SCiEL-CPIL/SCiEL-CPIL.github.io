"""Model-and-optimize approach: Directly optimize nodes and weights using smooth least-squares for weights and Nelder-Mead for nodes."""
import time
import numpy as np
from scipy.optimize import minimize, linprog
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

# Precompute high-accuracy Gauss-Legendre nodes/weights for reference integration
_REF_NODES, _REF_WEIGHTS = np.polynomial.legendre.leggauss(64)
_REF_NODES = 0.5 * (_REF_NODES + 1.0)
_REF_WEIGHTS = 0.5 * _REF_WEIGHTS

def _true_dL(z, w0, wa):
    if z == 0:
        return 0.0
    z_nodes = _REF_NODES * z
    a_vals = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(_REF_WEIGHTS / E_vals) * z
    return (1.0 + z) * integral

def _compute_test_data(nodes, test_cosmos, test_zs):
    num_tests = len(test_cosmos) * len(test_zs)
    F = np.zeros((num_tests, 8))
    T = np.zeros(num_tests)
    
    idx = 0
    for w0, wa in test_cosmos:
        for z in test_zs:
            if z == 0:
                T[idx] = 1.0
                F[idx, :] = 1.0
            else:
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = _E_cpl(a_vals, w0, wa)
                F[idx, :] = 1.0 / E_vals
                true_val = _true_dL(z, w0, wa)
                T[idx] = true_val / ((1.0 + z) * z)
            idx += 1
    return F, T

def _solve_weights_ls(F, T):
    """Solve for weights using least squares with positivity constraint."""
    n_vars = 8
    # Use non-negative least squares
    try:
        from scipy.optimize import nnls
        w, rnorm = nnls(F, T)
        w_sum = np.sum(w)
        if w_sum > 1e-10:
            w = w / w_sum
        else:
            w = np.ones(8) / 8.0
        return w
    except Exception:
        return np.ones(8) / 8.0

def _worst_rel_error(nodes, weights, test_cosmos, test_zs):
    """Compute worst relative error percentage."""
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            if z == 0:
                continue
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa)
            integral_approx = np.sum(weights / E_vals) * z
            approx_dL = (1.0 + z) * integral_approx
            true_dL = _true_dL(z, w0, wa)
            if abs(true_dL) > 1e-10:
                rel_err = abs(approx_dL - true_dL) / abs(true_dL) * 100.0
                max_err = max(max_err, rel_err)
    return max_err

def _softmax_error(nodes, weights, test_cosmos, test_zs, beta=50.0):
    """Smooth approximation to max error using log-sum-exp."""
    errors = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            if z == 0:
                continue
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa)
            integral_approx = np.sum(weights / E_vals) * z
            approx_dL = (1.0 + z) * integral_approx
            true_dL = _true_dL(z, w0, wa)
            if abs(true_dL) > 1e-10:
                rel_err = abs(approx_dL - true_dL) / abs(true_dL) * 100.0
                errors.append(rel_err)
    if not errors:
        return 0.0
    errors = np.array(errors)
    max_e = np.max(errors)
    if max_e < 1e-10:
        return 0.0
    # Log-sum-exp for smooth max
    scaled = beta * errors
    shifted = scaled - np.max(scaled)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + np.max(scaled)
    return log_sum_exp / beta

def _gauss_legendre_init():
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    idx = np.argsort(nodes)
    return nodes[idx]

def _encode_initial(nodes):
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    return -np.log(1.0/nodes - 1.0)

def _decode_params(params):
    nodes_raw = 1.0 / (1.0 + np.exp(-params))
    nodes = np.sort(nodes_raw)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    return nodes

def _joint_objective(params):
    if _time_left() < 2.0:
        return 1e10
    nodes = _decode_params(params)
    
    test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
        (-0.2, 0.2), (0.8, -0.8)
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    F, T = _compute_test_data(nodes, test_cosmos, test_zs)
    weights = _solve_weights_ls(F, T)
    
    # Use smooth max error as objective
    return _softmax_error(nodes, weights, test_cosmos, test_zs, beta=50.0)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    gl_nodes = _gauss_legendre_init()
    params_init = _encode_initial(gl_nodes)
    best_params = params_init.copy()
    best_score = _joint_objective(best_params)
    
    np.random.seed(42)
    restarts = 6
    for i in range(restarts):
        if _time_left() < 12.0:
            break
        if i == 0:
            init_p = params_init.copy()
        else:
            init_p = params_init + np.random.randn(8) * 0.8
        try:
            res = minimize(_joint_objective, init_p, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Final polish with tighter tolerances
    if _time_left() > 10.0:
        try:
            res_polish = minimize(_joint_objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                           'adaptive': True})
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass
    
    nodes = _decode_params(best_params)
    
    # Final weight optimization using broader test set
    final_test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
        (-0.2, 0.2), (0.8, -0.8), (0.1, 0.1), (-0.1, -0.1),
        (0.2, -0.5), (-0.8, 0.2)
    ]
    final_test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    F_final, T_final = _compute_test_data(nodes, final_test_cosmos, final_test_zs)
    weights = _solve_weights_ls(F_final, T_final)
    
    # Ensure weights are positive and sum to 1
    weights = np.abs(weights)
    w_sum = np.sum(weights)
    if w_sum < 1e-10:
        weights = np.ones(8) / 8.0
    else:
        weights = weights / w_sum
    
    # Verify constraints
    if not np.all(np.diff(nodes) > 0):
        # Fallback to Gauss-Legendre nodes
        nodes = _gauss_legendre_init()
        F_final, T_final = _compute_test_data(nodes, final_test_cosmos, final_test_zs)
        weights = _solve_weights_ls(F_final, T_final)
        weights = np.abs(weights)
        w_sum = np.sum(weights)
        if w_sum < 1e-10:
            weights = np.ones(8) / 8.0
        else:
            weights = weights / w_sum
    
    return {"nodes": nodes, "weights": weights}
