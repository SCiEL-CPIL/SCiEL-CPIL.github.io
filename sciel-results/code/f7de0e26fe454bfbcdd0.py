# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import lpmv
import math


def _build_quadrature(k=10):
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    return x, w, phi, dphi


def _compute_harmonics(x, phi, k=10):
    """
    Compute real orthonormal spherical harmonics Y_m for m = -k..k at points (x=cos theta, phi).
    Returns array of shape (n_x * n_phi, 2*k+1)
    """
    n_x = len(x)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_x, n_phi, n_m), dtype=np.float64)
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        log_num = math.lgamma(k + 1 - abs_m)
        log_den = math.lgamma(k + 1 + abs_m)
        N_m = math.sqrt((2 * k + 1) / (4 * np.pi) * math.exp(log_num - log_den))
        
        P = lpmv(abs_m, k, x)
        
        if m == 0:
            Y[:, :, 10] = N_m * P[:, np.newaxis]
        elif m > 0:
            Y[:, :, 10 + m] = math.sqrt(2) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            Y[:, :, 10 + m] = math.sqrt(2) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return Y.reshape(n_x * n_phi, n_m)


def _objective_normalized(c, Y, w, dphi):
    """
    Objective: ||f||_4 subject to ||f||_2 = 1.
    Since Y is orthonormal, ||f||_2^2 = sum(c^2).
    We normalize c to unit L2 norm, then compute ||f||_4.
    """
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-12:
        return 1e10
    c_unit = c / c_norm
    
    f = Y @ c_unit
    n_x = len(w)
    n_phi = Y.shape[0] // n_x
    weights_full = np.tile(w * dphi, n_phi)
    
    f2 = f * f
    f4 = f2 * f2
    
    norm4_sq = np.sum(weights_full * f4)
    
    if norm4_sq <= 0:
        return 0.0
        
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    return norm4


def rudin_problem10() -> np.ndarray:
    k = 10
    n_m = 2 * k + 1
    
    # Build quadrature
    x, w, phi, dphi = _build_quadrature(k)
    
    # Compute harmonics
    Y = _compute_harmonics(x, phi, k)
    
    # 1. Global Optimization using Differential Evolution
    bounds = [(-1.0, 1.0)] * n_m
    
    de_result = differential_evolution(
        _objective_normalized,
        bounds,
        args=(Y, w, dphi),
        seed=42,
        maxiter=300,
        popsize=15,
        tol=1e-8,
        polish=False
    )
    
    best_c_global = de_result.x
    
    # 2. Local Polish using L-BFGS-B
    def obj_local(c):
        return _objective_normalized(c, Y, w, dphi)
        
    res_local = minimize(
        obj_local,
        best_c_global,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
    )
    
    c_final = res_local.x
    
    # Validate and scale if necessary
    max_abs = np.max(np.abs(c_final))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 0:
            c_final = c_final / max_abs
            max_abs_new = np.max(np.abs(c_final))
            if max_abs_new < 1e-3:
                c_final = c_final * 1000
            elif max_abs_new > 1e5:
                c_final = c_final / 1000
        else:
            c_final = np.ones(n_m)
            
    return c_final.astype(np.float64)


# EVOLVE-BLOCK-END
