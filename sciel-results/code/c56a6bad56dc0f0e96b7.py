"""Spectral moment approach: minimize uncertainty bound via exact Gauss-Hermite quadrature
and global-to-local optimization strategy."""

import numpy as np
from scipy.optimize import minimize, dual_annealing, differential_evolution
from scipy.special import roots_hermite
import time

_DEADLINE = time.time() + 160

# Pre-compute Gauss-Hermite nodes and weights for exact integration
# We need exactness up to degree 28 (from x^4 * |P(x)^2| where P is deg 12)
# 2*N - 1 >= 28 => N >= 14.5 => N=15
_N_QUAD = 15
_x_nodes, _w_weights = roots_hermite(_N_QUAD)

def _make_g_coefficients(a0, a4, a8):
    """
    Given free coefficients a0, a4, a8, compute all polynomial coefficients
    P(x) = sum_{k=0}^{6} p_k x^{2k} for g(x) = P(x) e^{-x^2/2}.
    a12 is fixed to enforce g(0) = 0.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    return np.array([p0, p1, p2, p3, p4, p5, p6]), a12

def _compute_moments(p, nodes, weights):
    """
    Compute m_k = integral x^{2k} |P(x)|^2 e^{-x^2} dx for k=0,1,2
    using Gauss-Hermite quadrature (exact for polynomials).
    """
    # Evaluate P(x) at nodes
    # P(x) = p0 + p1 x^2 + ... + p6 x^12
    x2 = nodes**2
    # Vectorized polynomial evaluation
    # P(x) = sum p_i x^{2i}
    # We can use Horner's method or just summing
    # For small degree, summing is fine.
    P_vals = np.zeros_like(nodes)
    for i in range(7):
        P_vals += p[i] * (x2 ** i)
    
    # |P(x)|^2
    P_sq = P_vals * P_vals
    
    # m_k = sum w_i * x_i^{2k} * P_sq_i
    m = np.zeros(3)
    for k in range(3):
        m[k] = np.sum(weights * (nodes ** (2*k)) * P_sq)
        
    return m

def _compute_fourier_moments(p, nodes, weights):
    """
    Compute n_k = 2*pi * integral xi^{2k} |Q(xi)|^2 e^{-xi^2} dxi for k=0,1,2
    where Q(xi) = sum p_k (-1)^k xi^{2k}
    """
    # Q(xi) coefficients are q_i = p_i * (-1)^i
    q = p * ((-1.0) ** np.arange(7))
    
    # Evaluate Q(xi) at nodes
    x2 = nodes**2
    Q_vals = np.zeros_like(nodes)
    for i in range(7):
        Q_vals += q[i] * (x2 ** i)
        
    Q_sq = Q_vals * Q_vals
    
    n = np.zeros(3)
    for k in range(3):
        # Integral is 2*pi * sum w_i * xi^{2k} * Q_sq_i
        n[k] = 2.0 * np.pi * np.sum(weights * (nodes ** (2*k)) * Q_sq)
        
    return n

def _bound_value(params):
    """Compute the uncertainty bound for given parameters."""
    p, _ = _make_g_coefficients(params[0], params[1], params[2])
    
    m = _compute_moments(p, _x_nodes, _w_weights)
    n = _compute_fourier_moments(p, _x_nodes, _w_weights)
    
    if m[0] <= 0 or n[0] <= 0 or not np.all(np.isfinite(m)) or not np.all(np.isfinite(n)):
        return 1e18
    
    vf = m[1] / m[0]
    vF = n[1] / n[0]
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def construct():
    """
    Find optimal Hermite coefficients to minimize the uncertainty bound.
    Returns array of three coefficients [a0, a4, a8].
    """
    # Multiple diverse seed points for multi-start strategy
    seeds = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.02, 0.0001]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-1.0, 0.0, 0.0]),
    ]
    
    best_val = np.inf
    best_params = seeds[0].copy()
    
    # Evaluate all seeds
    for s in seeds:
        v = _bound_value(s)
        if v < best_val and np.isfinite(v):
            best_val = v
            best_params = s.copy()
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0)]
    
    # Stage 1: Differential Evolution for global search (more robust)
    if time.time() < _DEADLINE - 45:
        try:
            res_de = differential_evolution(
                _bound_value,
                bounds,
                seed=2024,
                maxiter=200,
                tol=1e-12,
                popsize=15,
                mutation=(0.3, 1.2),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if res_de.fun < best_val and np.isfinite(res_de.fun):
                best_val = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass
    
    # Stage 2: Dual Annealing from current best
    if time.time() < _DEADLINE - 25:
        try:
            res_da = dual_annealing(
                _bound_value, 
                bounds, 
                seed=2025, 
                maxiter=150, 
                initial_temp=5230.0, 
                restart_temp_ratio=2e-5, 
                visit=2.62, 
                maxiter_no_improvement=8,
                polish=False,
                x0=best_params
            )
            if res_da.fun < best_val and np.isfinite(res_da.fun):
                best_val = res_da.fun
                best_params = res_da.x.copy()
        except Exception:
            pass
    
    # Stage 3: L-BFGS-B polish
    if time.time() < _DEADLINE - 12:
        try:
            res_lbfgs = minimize(
                _bound_value, 
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 30000}
            )
            if res_lbfgs.fun < best_val and np.isfinite(res_lbfgs.fun):
                best_val = res_lbfgs.fun
                best_params = res_lbfgs.x.copy()
        except Exception:
            pass
    
    # Stage 4: Nelder-Mead final polish
    if time.time() < _DEADLINE - 5:
        try:
            res_nm = minimize(
                _bound_value, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_nm.fun < best_val and np.isfinite(res_nm.fun):
                best_val = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
    
    if best_params is None or not np.all(np.isfinite(best_params)):
        return seeds[0]
        
    return np.array(best_params)
