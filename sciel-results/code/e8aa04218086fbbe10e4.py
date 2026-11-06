"""Improved program: vectorized reference + smooth-max objective for 8-node quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry."""
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


# Pre-compute 128-point Gauss-Legendre for exact reference
_GX, _GW = np.polynomial.legendre.leggauss(128)


def _exact_comoving_vec(z_arr, w0, wa):
    """Vectorized exact comoving distance for array of z values."""
    z_arr = np.asarray(z_arr)
    # For each z, exact = 0.5*z * sum(GW / E(0.5*z*(GX+1)))
    # Shape: (len(z_arr), 128)
    z_mid = 0.5 * z_arr[:, None] * (_GX[None, :] + 1.0)
    E_vals = _cpl_E(z_mid, w0, wa)
    return 0.5 * z_arr * np.sum(_GW[None, :] / E_vals, axis=1)


def _quad_comoving_vec(z_arr, w0, wa, nodes, weights):
    """Vectorized quadrature comoving distance for array of z values."""
    z_arr = np.asarray(z_arr)
    # z * sum(weights / E(z*nodes))
    z_mid = z_arr[:, None] * nodes[None, :]
    E_vals = _cpl_E(z_mid, w0, wa)
    return z_arr * np.sum(weights[None, :] / E_vals, axis=1)


def _precompute_reference(w0_vals, wa_vals, z_vals):
    """Pre-compute exact reference values for all (w0, wa, z) combinations."""
    ref_dict = {}
    for w0 in w0_vals:
        for wa in wa_vals:
            exact = _exact_comoving_vec(z_vals, w0, wa)
            ref_dict[(w0, wa)] = exact
    return ref_dict


def _worst_error_fast(nodes, weights, w0_vals, wa_vals, z_vals, ref_dict):
    """Compute worst relative error over ensemble using precomputed references."""
    worst = 0.0
    for w0 in w0_vals:
        for wa in wa_vals:
            exact = ref_dict[(w0, wa)]
            quad = _quad_comoving_vec(z_vals, w0, wa, nodes, weights)
            rel_errs = np.abs(quad - exact) / np.maximum(np.abs(exact), 1e-30)
            worst = max(worst, float(np.max(rel_errs)))
    return worst


def _smooth_max(errors, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(errors)
    if m < 1e-30:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (errors - m)))) / beta


def _smooth_objective(nodes, weights, w0_vals, wa_vals, z_vals, ref_dict, beta=50.0):
    """Smooth surrogate objective: soft-max of relative errors."""
    all_errors = []
    for w0 in w0_vals:
        for wa in wa_vals:
            exact = ref_dict[(w0, wa)]
            quad = _quad_comoving_vec(z_vals, w0, wa, nodes, weights)
            rel_errs = np.abs(quad - exact) / np.maximum(np.abs(exact), 1e-30)
            all_errors.append(rel_errs)
    all_errors = np.concatenate(all_errors)
    return _smooth_max(all_errors, beta)


def _optimize_fast(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Optimize nodes and weights with smooth objective and vectorized eval."""
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    
    # Precompute reference
    ref_dict = _precompute_reference(w0_vals, wa_vals, z_vals)
    
    best_err = _worst_error_fast(best_nodes, best_weights, w0_vals, wa_vals, z_vals, ref_dict)
    
    def _params_to_nw(params):
        x = params[:n]
        y = params[n:]
        # Nodes: use cumulative exp to ensure (0,1) increasing
        ex = np.exp(x - np.max(x))
        cumsum = np.cumsum(ex)
        nodes_p = cumsum / cumsum[-1]
        eps = 1e-12
        nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
        for i in range(1, n):
            if nodes_p[i] <= nodes_p[i-1]:
                nodes_p[i] = nodes_p[i-1] + eps
        nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
        # Weights: exp parameterization
        ey = np.exp(y - np.max(y))
        weights_p = ey / ey.sum()
        return nodes_p, weights_p
    
    def _objective_smooth(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        return _smooth_objective(nodes_p, weights_p, w0_vals, wa_vals, z_vals, ref_dict, beta=50.0)
    
    def _objective_exact(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        return _worst_error_fast(nodes_p, weights_p, w0_vals, wa_vals, z_vals, ref_dict)
    
    # Initial params
    x_init = np.log(np.maximum(best_nodes, 1e-15))
    y_init = np.log(np.maximum(best_weights, 1e-15))
    p0 = np.concatenate([x_init, y_init])
    
    t_start = time.time()
    
    # Phase 1: Multi-start Nelder-Mead with smooth objective
    rng = np.random.RandomState(42)
    for restart in range(6):
        if not _deadline_ok() or (time.time() - t_start) > time_budget * 0.5:
            break
        
        if restart == 0:
            p_cur = p0.copy()
        else:
            p_cur = p0 + rng.normal(0, 0.15, size=2*n)
        
        res = minimize(
            _objective_smooth, p_cur,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True,
                     'maxfev': 200}
        )
        if _deadline_ok():
            np_, wp_ = _params_to_nw(res.x)
            err_ = _worst_error_fast(np_, wp_, w0_vals, wa_vals, z_vals, ref_dict)
            if err_ < best_err:
                best_nodes, best_weights, best_err = np_, wp_, err_
                p0 = res.x.copy()
    
    # Phase 2: L-BFGS-B with smooth objective (gradient-friendly)
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.7:
        res2 = minimize(
            _objective_smooth, p0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-14, 'maxfun': 300}
        )
        if _deadline_ok():
            np2, wp2 = _params_to_nw(res2.x)
            err2 = _worst_error_fast(np2, wp2, w0_vals, wa_vals, z_vals, ref_dict)
            if err2 < best_err:
                best_nodes, best_weights, best_err = np2, wp2, err2
                p0 = res2.x.copy()
    
    # Phase 3: Fine-tune with exact objective via Nelder-Mead
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.85:
        res3 = minimize(
            _objective_exact, p0,
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True,
                     'maxfev': 120}
        )
        if _deadline_ok():
            np3, wp3 = _params_to_nw(res3.x)
            err3 = _worst_error_fast(np3, wp3, w0_vals, wa_vals, z_vals, ref_dict)
            if err3 < best_err:
                best_nodes, best_weights, best_err = np3, wp3, err3
    
    return best_nodes, best_weights


def construct():
    n = 8
    # Start from Gauss-Legendre nodes/weights on [0,1]
    xs, ws = np.polynomial.legendre.leggauss(n)
    nodes_init = 0.5 * (xs + 1.0)
    weights_init = 0.5 * ws
    nodes_init = np.clip(nodes_init, 1e-15, 1.0 - 1e-15)
    for i in range(1, n):
        if nodes_init[i] <= nodes_init[i-1]:
            nodes_init[i] = nodes_init[i-1] + 1e-15
    nodes_init[-1] = min(nodes_init[-1], 1.0 - 1e-15)
    weights_init = weights_init / weights_init.sum()
    
    # Focused ensemble: emphasize regions where error is likely worst
    w0_vals = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5]
    wa_vals = [-0.5, -0.3, 0.0, 0.3, 0.5, 0.8]
    z_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    time_budget = 80.0
    
    try:
        nodes, weights = _optimize_fast(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget)
    except Exception:
        nodes, weights = nodes_init, weights_init
    
    # Final cleanup
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    nodes[-1] = min(nodes[-1], 1.0 - 1e-12)
    weights = np.maximum(weights, 1e-15)
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
