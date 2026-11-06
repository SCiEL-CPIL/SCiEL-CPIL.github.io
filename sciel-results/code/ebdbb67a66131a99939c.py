"""Improved program: fast optimization of 8-node universal quadrature."""
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


def _exact_comoving_fast(z, w0, wa):
    """Exact comoving distance using adaptive Simpson or Gauss-Kronrod."""
    if z <= 0:
        return 0.0
    # Use 64-point Gauss-Legendre which is very accurate for smooth integrands
    xs, ws = np.polynomial.legendre.leggauss(64)
    z_vals = 0.5 * z * (xs + 1.0)
    E_vals = _cpl_E(z_vals, w0, wa)
    return 0.5 * z * np.sum(ws / E_vals)


def _quad_comoving(z, w0, wa, nodes, weights):
    """Quadrature comoving distance."""
    if z <= 0:
        return 0.0
    z_vals = z * nodes
    E_vals = _cpl_E(z_vals, w0, wa)
    return z * np.sum(weights / E_vals)


def _worst_error_fast(nodes, weights, w0_vals, wa_vals, z_vals):
    """Compute worst relative error over ensemble."""
    worst = 0.0
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                exact = _exact_comoving_fast(z, w0, wa)
                quad = _quad_comoving(z, w0, wa, nodes, weights)
                rel_err = abs(quad - exact) / max(abs(exact), 1e-30)
                worst = max(worst, rel_err)
    return worst


def _optimize_fast(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Optimize nodes and weights with efficient objective."""
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error_fast(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
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
    
    def _objective(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        return _worst_error_fast(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
    
    # Initial params
    x_init = np.log(np.maximum(best_nodes, 1e-15))
    y_init = np.log(np.maximum(best_weights, 1e-15))
    p0 = np.concatenate([x_init, y_init])
    
    t_start = time.time()
    
    # Multi-start Nelder-Mead
    rng = np.random.RandomState(42)
    for restart in range(5):
        if not _deadline_ok() or (time.time() - t_start) > time_budget * 0.7:
            break
        
        if restart == 0:
            p_cur = p0.copy()
        else:
            # Perturb initial point
            p_cur = p0 + rng.normal(0, 0.1, size=2*n)
        
        res = minimize(
            _objective, p_cur,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True,
                     'maxfev': 150}
        )
        if _deadline_ok():
            np_, wp_ = _params_to_nw(res.x)
            err_ = _worst_error_fast(np_, wp_, w0_vals, wa_vals, z_vals)
            if err_ < best_err:
                best_nodes, best_weights, best_err = np_, wp_, err_
                p0 = res.x.copy()  # Update starting point for next restart
    
    # Final L-BFGS-B polish if time permits
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.8:
        res2 = minimize(
            _objective, p0,
            method='L-BFGS-B',
            options={'maxiter': 80, 'gtol': 1e-12, 'maxfun': 100}
        )
        if _deadline_ok():
            np2, wp2 = _params_to_nw(res2.x)
            err2 = _worst_error_fast(np2, wp2, w0_vals, wa_vals, z_vals)
            if err2 < best_err:
                best_nodes, best_weights, best_err = np2, wp2, err2
    
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
