"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _e_cpl(a, w0, wa):
    """CPL E(a) = H(a)/H0 for spatially flat universe."""
    if abs(w0 + wa) < 1e-10:
        return a ** (-3.0 * (1.0 + w0))
    return a ** (-3.0 * (1.0 + w0 + wa)) * ((1.0 + w0 + wa) / (1.0 + w0)) * (1.0 / (1.0 + a)) ** wa


def _integrand(t, w0, wa):
    """Integrand for luminosity distance: (1+t)*t / E(1/(1+t))"""
    a = 1.0 / (1.0 + t)
    return (1.0 + t) * t / _e_cpl(a, w0, wa)


def _reference_integral(z, w0, wa, n_sub=1000):
    """High-accuracy reference integral using dense trapezoidal rule."""
    t = np.linspace(0, z, n_sub + 1)
    f = np.array([_integrand(ti, w0, wa) for ti in t])
    return np.trapz(f, t)


def _quadrature_error(nodes, weights, z, w0, wa):
    """Relative error of quadrature vs reference."""
    a_vals = 1.0 / (1.0 + z * nodes)
    q = (1.0 + z) * z * np.sum(weights / _e_cpl(a_vals, w0, wa))
    ref = _reference_integral(z, w0, wa)
    if abs(ref) < 1e-15:
        return 0.0
    return abs(q - ref) / abs(ref) * 100.0  # percent


def _worst_case_error(nodes, weights):
    """Worst-case relative error across representative CPL models and redshifts."""
    w0_vals = [-1.2, -1.0, -0.8, -0.7, -0.6, -0.5, 0.0, 0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    z_vals = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    max_err = 0.0
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                err = _quadrature_error(nodes, weights, z, w0, wa)
                if err > max_err:
                    max_err = err
    return max_err


def _objective(params):
    """Objective: worst-case relative error with penalty for invalid nodes."""
    nodes_raw = params[:8]
    weights_raw = params[8:]
    
    # Transform to ensure nodes in (0,1) strictly increasing
    # Use cumulative sum of positive values
    diffs = np.exp(nodes_raw)  # positive
    nodes = np.cumsum(diffs)
    nodes = nodes / nodes[-1]  # normalize to [0,1]
    
    # Ensure strictly positive weights
    weights = np.exp(weights_raw)
    weights = weights / weights.sum()
    
    # Check strict increase
    if np.any(np.diff(nodes) <= 0):
        return 1e6
    
    return _worst_case_error(nodes, weights)


def construct():
    deadline = time.time() + 80.0  # safety margin
    
    # Start from Gauss-Legendre-like nodes
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (x_gl + 1.0)  # map to [0,1]
    nodes0 = np.sort(nodes0)
    
    # Initial parameterization
    diffs0 = np.diff(np.concatenate([[0], nodes0]))
    nodes0_params = np.log(diffs0)
    weights0_params = np.log(np.full(8, 1.0/8))
    
    x0 = np.concatenate([nodes0_params, weights0_params])
    
    # Optimize
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # L-BFGS-B
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-10})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() > deadline:
        pass  # use best so far
    
    # Powell refinement
    try:
        res2 = minimize(_objective, best_x, method='Powell',
                        options={'maxiter': 100, 'xtol': 1e-12})
        if res2.fun < best_obj:
            best_obj = res2.fun
            best_x = res2.x
    except Exception:
        pass
    
    # Extract final nodes and weights
    nodes_raw = best_x[:8]
    weights_raw = best_x[8:]
    
    diffs = np.exp(nodes_raw)
    nodes = np.cumsum(diffs)
    nodes = nodes / nodes[-1]
    
    weights = np.exp(weights_raw)
    weights = weights / weights.sum()
    
    # Final validation
    if np.any(nodes <= 0) or np.any(nodes >= 1) or np.any(np.diff(nodes) <= 0):
        # Fallback to midpoint rule
        nodes = (np.arange(8) + 0.5) / 8.0
        weights = np.full(8, 1.0 / 8.0)
    if np.any(weights <= 0):
        weights = np.full(8, 1.0 / 8.0)
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
