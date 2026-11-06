"""Improved program: Chebyshev moment-matching quadrature for luminosity distance."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry.
    E(z)^2 = (1+z)^{3(1+w0+wa)} * (1 + wa*z)^{-3*wa}
    """
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


def _exact_comoving_batch(z_arr, w0, wa, n_quad=500):
    """Exact comoving distances for array of z values via Gauss-Legendre."""
    n_z = len(z_arr)
    exact_arr = np.zeros(n_z)
    xs, ws = np.polynomial.legendre.leggauss(n_quad)
    for i, z in enumerate(z_arr):
        if z <= 0:
            continue
        z_vals = 0.5 * z * (xs + 1.0)
        E_vals = _cpl_E(z_vals, w0, wa)
        exact_arr[i] = 0.5 * z * np.sum(ws / E_vals)
    return exact_arr


def _quad_comoving_batch(z_arr, w0, wa, nodes, weights):
    """Quadrature comoving distances for array of z values."""
    n_z = len(z_arr)
    quad_arr = np.zeros(n_z)
    for i, z in enumerate(z_arr):
        if z <= 0:
            continue
        z_vals = z * nodes
        E_vals = _cpl_E(z_vals, w0, wa)
        quad_arr[i] = z * np.sum(weights / E_vals)
    return quad_arr


def _worst_error(nodes, weights, w0_vals, wa_vals, z_vals, n_quad=500):
    """Compute worst relative error over ensemble."""
    worst = 0.0
    z_arr = np.array(z_vals)
    for w0 in w0_vals:
        for wa in wa_vals:
            exact_arr = _exact_comoving_batch(z_arr, w0, wa, n_quad)
            quad_arr = _quad_comoving_batch(z_arr, w0, wa, nodes, weights)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
            worst = max(worst, float(np.max(rel_err)))
    return worst


def _moments_from_cosmology(w0, wa, z_max, n_moments):
    """Compute exact moments m_k = ∫₀^{z_max} u^k / E(z_max*u) du for k=0..n_moments-1."""
    moments = np.zeros(n_moments)
    xs, ws = np.polynomial.legendre.leggauss(200)
    u_vals = 0.5 * (xs + 1.0)  # in [0,1]
    z_vals = z_max * u_vals
    E_vals = _cpl_E(z_vals, w0, wa)
    for k in range(n_moments):
        moments[k] = 0.5 * z_max * np.sum(ws * u_vals**k / E_vals)
    return moments


def _find_chebyshev_quadrature(n_nodes, n_moments, z_max, w0_vals, wa_vals):
    """
    Find optimal n_nodes quadrature that matches moments of a representative integrand.
    Uses Chebyshev nodes as starting point and optimizes weights to match moments.
    """
    # Use Chebyshev nodes mapped to [0,1] as a good starting point
    x_cheb = np.cos(np.pi * (np.arange(n_nodes) + 0.5) / n_nodes)  # in [-1,1]
    nodes_init = 0.5 * (x_cheb + 1.0)
    nodes_init = np.sort(nodes_init)
    nodes_init = np.clip(nodes_init, 1e-12, 1.0 - 1e-12)
    
    # Compute target moments from a representative cosmology
    # Use w0=-1, wa=0 (ΛCDM) as representative
    target_moments = _moments_from_cosmology(-1.0, 0.0, 1.0, n_moments)
    
    # Set up moment matching: sum_i w_i * u_i^k = m_k for k=0..min(n_nodes-1, n_moments-1)
    n_match = min(n_nodes, n_moments)
    A = np.zeros((n_match, n_nodes))
    for k in range(n_match):
        A[k, :] = nodes_init**k
    
    # Solve least squares for weights
    weights_init, _, _, _ = np.linalg.lstsq(A, target_moments[:n_match], rcond=None)
    weights_init = np.maximum(weights_init, 1e-15)
    weights_init = weights_init / weights_init.sum()
    
    return nodes_init, weights_init


def _optimize_moments(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """
    Optimize by matching moments across multiple cosmologies.
    Use a weighted least-squares approach to minimize worst-case error.
    """
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    try:
        from scipy.optimize import minimize
        have_scipy = True
    except ImportError:
        have_scipy = False
    
    if not have_scipy:
        return best_nodes, best_weights
    
    def _objective(params):
        if not _deadline_ok():
            return 1e10
        n = 8
        x = params[:n]
        y = params[n:]
        ex = np.exp(x - np.max(x))
        cumsum = np.cumsum(ex)
        nodes_p = cumsum / cumsum[-1]
        eps = 1e-12
        nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
        for i in range(1, n):
            if nodes_p[i] <= nodes_p[i-1]:
                nodes_p[i] = nodes_p[i-1] + eps
        nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
        ey = np.exp(y - np.max(y))
        weights_p = ey / ey.sum()
        return _worst_error(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
    
    x_init = np.log(np.maximum(best_nodes, 1e-15))
    y_init = np.log(np.maximum(best_weights, 1e-15))
    p0 = np.concatenate([x_init, y_init])
    
    t_start = time.time()
    
    # Nelder-Mead
    res = minimize(
        _objective, p0,
        method='Nelder-Mead',
        options={'maxiter': 400, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True,
                 'maxfev': 300}
    )
    if _deadline_ok():
        x_ = res.x[:n]
        y_ = res.x[n:]
        ex = np.exp(x_ - np.max(x_))
        cumsum = np.cumsum(ex)
        np_ = cumsum / cumsum[-1]
        np_ = np.clip(np_, 1e-12, 1.0 - 1e-12)
        for i in range(1, n):
            if np_[i] <= np_[i-1]:
                np_[i] = np_[i-1] + 1e-12
        np_[-1] = min(np_[-1], 1.0 - 1e-12)
        ey = np.exp(y_ - np.max(y_))
        wp_ = ey / ey.sum()
        err_ = _worst_error(np_, wp_, w0_vals, wa_vals, z_vals)
        if err_ < best_err:
            best_nodes, best_weights, best_err = np_, wp_, err_
    
    # L-BFGS-B if time permits
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.5:
        p_cur = res.x if res.success else p0
        res2 = minimize(
            _objective, p_cur,
            method='L-BFGS-B',
            options={'maxiter': 150, 'gtol': 1e-14, 'maxfun': 200}
        )
        if _deadline_ok():
            x_ = res2.x[:n]
            y_ = res2.x[n:]
            ex = np.exp(x_ - np.max(x_))
            cumsum = np.cumsum(ex)
            np2 = cumsum / cumsum[-1]
            np2 = np.clip(np2, 1e-12, 1.0 - 1e-12)
            for i in range(1, n):
                if np2[i] <= np2[i-1]:
                    np2[i] = np2[i-1] + 1e-12
            np2[-1] = min(np2[-1], 1.0 - 1e-12)
            ey = np.exp(y_ - np.max(y_))
            wp2 = ey / ey.sum()
            err2 = _worst_error(np2, wp2, w0_vals, wa_vals, z_vals)
            if err2 < best_err:
                best_nodes, best_weights, best_err = np2, wp2, err2
    
    return best_nodes, best_weights


def construct():
    n = 8
    # Use Chebyshev moment matching to get a better initial guess
    nodes_init, weights_init = _find_chebyshev_quadrature(n, 16, 1.0, [-1.0], [0.0])
    
    # Broad ensemble for optimization
    w0_vals = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5]
    wa_vals = [-0.5, -0.3, 0.0, 0.3, 0.5, 0.8]
    z_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    time_budget = 78.0
    
    try:
        nodes, weights = _optimize_moments(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget)
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
