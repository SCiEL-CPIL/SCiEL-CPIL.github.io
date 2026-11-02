"""Analytical gradient-based optimization for Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import time

_DEADLINE = 200  # Safety margin before 210s kill

def _hermite_h(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 7:
        return 128*x**7 - 1344*x**5 + 3360*x**3 - 1680*x
    elif n == 8:
        return 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
    elif n == 11:
        return (2048*x**11 - 77824*x**9 + 887040*x**7
                - 4084080*x**5 + 8316000*x**3 - 6652800*x)
    elif n == 12:
        return (4096*x**12 - 147456*x**10 + 2580480*x**8
                - 20321280*x**6 + 81729600*x**4
                - 172972800*x**2 + 103219200)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

# Precompute high-precision quadrature data
_N_QUAD = 500
_x_quad, _w_quad = roots_hermite(_N_QUAD)
_H0 = _hermite_h(0, _x_quad)
_H4 = _hermite_h(4, _x_quad)
_H8 = _hermite_h(8, _x_quad)
_H12 = _hermite_h(12, _x_quad)
_dH0 = np.zeros_like(_x_quad)
_dH4 = 8 * _hermite_h(3, _x_quad)
_dH8 = 16 * _hermite_h(7, _x_quad)
_dH12 = 24 * _hermite_h(11, _x_quad)

# Values at x=0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 40320.0
_H12_0 = 103219200.0

def _compute_bound_and_grad(coeffs):
    """
    Compute the upper bound on C4 and its gradient given coefficients [a0, a4, a8].
    Returns (bound, gradient).
    """
    a0, a4, a8 = coeffs
    a = np.array([a0, a4, a8])
    
    # d is linear in a: d = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    # So d is a linear function of the parameters
    H0_0, H4_0, H8_0, H12_0 = _H0_0, _H4_0, _H8_0, _H12_0
    d = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # P = a0*H0 + a4*H4 + a8*H8 + d*H12
    # Since d is linear in a, P is linear in a
    # P = H0*a0 + H4*a4 + H8*a8 + H12*d
    #   = H0*a0 + H4*a4 + H8*a8 + H12*(-(H0_0*a0 + H4_0*a4 + H8_0*a8)/H12_0)
    #   = a0*(H0 - H0_0*H12/H12_0) + a4*(H4 - H4_0*H12/H12_0) + a8*(H8 - H8_0*H12/H12_0)
    
    H0_eff = _H0 - H0_0 * _H12 / H12_0
    H4_eff = _H4 - H4_0 * _H12 / H12_0
    H8_eff = _H8 - H8_0 * _H12 / H12_0
    
    P = a0 * H0_eff + a4 * H4_eff + a8 * H8_eff
    
    # dP/dx = a0*dH0_eff + a4*dH4_eff + a8*dH8_eff
    # dH0_eff = dH0 - H0_0*dH12/H12_0 = 0 - 1*dH12/H12_0
    # dH4_eff = dH4 - H4_0*dH12/H12_0
    # dH8_eff = dH8 - H8_0*dH12/H12_0
    dH0_eff = _dH0 - H0_0 * _dH12 / H12_0
    dH4_eff = _dH4 - H4_0 * _dH12 / H12_0
    dH8_eff = _dH8 - H8_0 * _dH12 / H12_0
    
    dP = a0 * dH0_eff + a4 * dH4_eff + a8 * dH8_eff
    
    P2 = P * P
    norm2 = np.sum(_w_quad * P2)
    
    if norm2 <= 1e-15 or not np.isfinite(norm2):
        return 1e10, np.zeros(3)
    
    # var_x = sum(w * x^2 * P^2) / norm2
    var_x = np.sum(_w_quad * _x_quad**2 * P2) / norm2
    
    # deriv_term = dP - x*P
    deriv_term = dP - _x_quad * P
    deriv_term_sq = deriv_term * deriv_term
    norm_deriv2 = np.sum(_w_quad * deriv_term_sq)
    
    if not np.isfinite(norm_deriv2) or norm_deriv2 < 0:
        return 1e10, np.zeros(3)
    
    var_k = norm_deriv2 / norm2
    
    bound = var_x * var_k
    
    # Now compute gradient analytically
    # bound = var_x * var_k
    # var_x = N1 / norm2 where N1 = sum(w*x^2*P^2)
    # var_k = N2 / norm2 where N2 = sum(w*(dP-x*P)^2)
    
    # d(norm2)/da_i = 2 * sum(w * P * dP_i) where dP_i = dP/da_i
    # dP/da0 = H0_eff, dP/da4 = H4_eff, dP/da8 = H8_eff
    # d(dP)/da0 = dH0_eff, d(dP)/da4 = dH4_eff, d(dP)/da8 = dH8_eff
    
    dP_da0 = H0_eff
    dP_da4 = H4_eff
    dP_da8 = H8_eff
    ddP_da0 = dH0_eff
    ddP_da4 = dH4_eff
    ddP_da8 = dH8_eff
    
    # d(norm2)/da_i = 2 * sum(w * P * dP_da_i)
    d_norm2_da0 = 2.0 * np.sum(_w_quad * P * dP_da0)
    d_norm2_da4 = 2.0 * np.sum(_w_quad * P * dP_da4)
    d_norm2_da8 = 2.0 * np.sum(_w_quad * P * dP_da8)
    
    # d(N1)/da_i where N1 = sum(w*x^2*P^2)
    d_N1_da0 = 2.0 * np.sum(_w_quad * _x_quad**2 * P * dP_da0)
    d_N1_da4 = 2.0 * np.sum(_w_quad * _x_quad**2 * P * dP_da4)
    d_N1_da8 = 2.0 * np.sum(_w_quad * _x_quad**2 * P * dP_da8)
    
    # d(N2)/da_i where N2 = sum(w*(dP - x*P)^2)
    # Let Q = dP - x*P
    # dQ/da_i = ddP_da_i - x * dP_da_i
    Q = deriv_term
    dQ_da0 = ddP_da0 - _x_quad * dP_da0
    dQ_da4 = ddP_da4 - _x_quad * dP_da4
    dQ_da8 = ddP_da8 - _x_quad * dP_da8
    
    d_N2_da0 = 2.0 * np.sum(_w_quad * Q * dQ_da0)
    d_N2_da4 = 2.0 * np.sum(_w_quad * Q * dQ_da4)
    d_N2_da8 = 2.0 * np.sum(_w_quad * Q * dQ_da8)
    
    # var_x = N1/norm2, var_k = N2/norm2
    # d(var_x)/da_i = (d_N1_da_i * norm2 - N1 * d_norm2_da_i) / norm2^2
    # d(var_k)/da_i = (d_N2_da_i * norm2 - N2 * d_norm2_da_i) / norm2^2
    # d(bound)/da_i = var_k * d(var_x)/da_i + var_x * d(var_k)/da_i
    
    N1 = np.sum(_w_quad * _x_quad**2 * P2)
    N2 = norm_deriv2
    
    d_var_x_da0 = (d_N1_da0 * norm2 - N1 * d_norm2_da0) / (norm2 * norm2)
    d_var_x_da4 = (d_N1_da4 * norm2 - N1 * d_norm2_da4) / (norm2 * norm2)
    d_var_x_da8 = (d_N1_da8 * norm2 - N1 * d_norm2_da8) / (norm2 * norm2)
    
    d_var_k_da0 = (d_N2_da0 * norm2 - N2 * d_norm2_da0) / (norm2 * norm2)
    d_var_k_da4 = (d_N2_da4 * norm2 - N2 * d_norm2_da4) / (norm2 * norm2)
    d_var_k_da8 = (d_N2_da8 * norm2 - N2 * d_norm2_da8) / (norm2 * norm2)
    
    grad0 = var_k * d_var_x_da0 + var_x * d_var_k_da0
    grad4 = var_k * d_var_x_da4 + var_x * d_var_k_da4
    grad8 = var_k * d_var_x_da8 + var_x * d_var_k_da8
    
    gradient = np.array([grad0, grad4, grad8])
    
    if not np.all(np.isfinite(gradient)):
        gradient = np.zeros(3)
    
    return bound, gradient

def _objective(coeffs):
    """Objective function for L-BFGS-B."""
    if time.time() > _DEADLINE:
        return 1e10
    bound, _ = _compute_bound_and_grad(coeffs)
    return bound

def _objective_with_grad(coeffs):
    """Objective function with gradient for L-BFGS-B."""
    if time.time() > _DEADLINE:
        return 1e10, np.zeros(3)
    return _compute_bound_and_grad(coeffs)

def construct():
    start_time = time.time()
    
    # Seed point
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_result = seed.copy()
    best_score = _compute_bound_and_grad(seed)[0]
    
    # Multi-start L-BFGS-B with analytical gradients
    # The coefficients have natural scales based on the problem structure
    # a0 ~ O(100), a4 ~ O(0.01), a8 ~ O(1e-4)
    
    start_points = [
        seed,
        # Perturbations around seed at various scales
        seed + np.array([5.0, 0.005, 0.0005]),
        seed - np.array([5.0, 0.005, 0.0005]),
        seed + np.array([20.0, 0.02, 0.002]),
        seed - np.array([20.0, 0.02, 0.002]),
        # Different scale explorations
        np.array([-100.0, 0.04, 0.0001]),
        np.array([-130.0, 0.004, 0.0003]),
        np.array([-90.0, 0.008, 0.0002]),
        np.array([-120.0, 0.012, 0.00015]),
        np.array([-113.0, 0.02, 0.0003]),
        np.array([-113.0, 0.003, 0.0001]),
        np.array([-113.0, 0.01, 0.00005]),
    ]
    
    # Run L-BFGS-B from each start point
    for i, start in enumerate(start_points):
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(
                _objective_with_grad,
                start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Second round: finer local search around the best found
    if time.time() < _DEADLINE - 10:
        local_starts = [
            best_result,
            best_result + np.array([1.0, 0.001, 0.0001]),
            best_result - np.array([1.0, 0.001, 0.0001]),
            best_result + np.array([0.1, 0.0001, 0.00001]),
            best_result - np.array([0.1, 0.0001, 0.00001]),
        ]
        for start in local_starts:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res = minimize(
                    _objective_with_grad,
                    start,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxfun': 5000}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
    
    # Final: very tight local polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(
                _objective_with_grad,
                best_result,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxfun': 10000}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_result)

# EVOLVE-BLOCK-END
