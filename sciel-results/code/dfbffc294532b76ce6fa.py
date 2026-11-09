# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

# Precompute spherical harmonic basis functions on the quadrature grid
# 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
def _setup_grid():
    """Set up the quadrature grid and precompute Y_m values."""
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in x = cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x in [-1, 1], w are weights for integral over [-1, 1]
    # For surface integral: dOmega = dphi * d(cos(theta)), so weight = w_i * (2*pi/n_phi)
    
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Create meshgrid
    X, PHI = np.meshgrid(x, phi, indexing='ij')  # X shape (n_theta, n_phi)
    
    # Surface measure weights: w_i * (2*pi/n_phi)
    phi_weight = 2 * np.pi / n_phi
    # Total weight for each (theta_i, phi_j) point
    weights = np.outer(w, np.full(n_phi, phi_weight))  # shape (n_theta, n_phi)
    
    # Precompute Y_m for m = -10..10
    # Y_0 = N_0 * P_10^0(cos theta)
    # Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^|m|(cos theta) * sin(|m|*phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    
    Y = np.zeros((21, n_theta, n_phi), dtype=np.float64)
    
    from math import factorial, pi, sqrt
    
    for m_idx in range(21):
        m = m_idx - 10  # m ranges from -10 to 10
        abs_m = abs(m)
        
        # N_m = sqrt(21/(4*pi) * (10-abs_m)!/(10+abs_m)!)
        N_m = sqrt(21.0 / (4.0 * pi) * factorial(10 - abs_m) / factorial(10 + abs_m))
        
        # P_10^m(cos theta) with Condon-Shortley phase (scipy's lpmv includes it)
        P = lpmv(abs_m, 10, X)  # shape (n_theta, n_phi) but X only depends on theta
        
        if m == 0:
            Y[m_idx] = N_m * P
        elif m > 0:
            Y[m_idx] = sqrt(2.0) * N_m * P * np.cos(m * PHI)
        else:
            Y[m_idx] = sqrt(2.0) * N_m * P * np.sin(abs_m * PHI)
    
    return Y, weights, n_theta, n_phi

# Global precomputation
_Y, _weights, _n_theta, _n_phi = _setup_grid()
_flat_weights = _weights.ravel()
_Y_flat = _Y.reshape(21, -1)  # shape (21, n_theta*n_phi)


def _compute_ratio(coeffs):
    """Compute R = ||f||_4 / ||f||_2 using precomputed grid."""
    # f = sum_m c_m * Y_m
    f = _Y_flat @ coeffs  # shape (n_points,)
    
    f2 = np.sum(f * f * _flat_weights)
    f4 = np.sum(f * f * f * f * _flat_weights)
    
    if f2 <= 0:
        return np.inf
    
    norm2 = np.sqrt(f2)
    norm4 = np.sqrt(np.sqrt(f4))  # ||f||_4 = (integral |f|^4)^(1/4)
    
    return norm4 / norm2


def _objective_and_grad(coeffs):
    """Objective: R^2 to avoid sqrt, and its gradient."""
    f = _Y_flat @ coeffs
    f2 = np.sum(f * f * _flat_weights)
    f4 = np.sum(f * f * f * f * _flat_weights)
    
    if f2 <= 0:
        return 1e10, np.zeros_like(coeffs)
    
    # R = (f4^(1/4)) / (f2^(1/2))
    # R^2 = f4^(1/2) / f2
    # Let g = R^2 = sqrt(f4) / f2
    # dg/df4 = (1/(2*sqrt(f4))) * (1/f2)
    # dg/df2 = -sqrt(f4) / f2^2
    # df4/dc = 4 * f^3 * Y * w
    # df2/dc = 2 * f * Y * w
    
    sqrt_f4 = np.sqrt(f4)
    g = sqrt_f4 / f2
    
    # Gradient
    df4_dc = 4.0 * (f ** 3)[:, None] * _Y_flat * _flat_weights[:, None]
    df2_dc = 2.0 * f[:, None] * _Y_flat * _flat_weights[:, None]
    
    dg_dc = (1.0 / (2.0 * sqrt_f4)) * (df4_dc / f2) - (sqrt_f4 / (f2 ** 2)) * df2_dc
    grad = np.sum(dg_dc, axis=0)
    
    return g, grad


def _objective(coeffs):
    return _objective_and_grad(coeffs)[0]


def _grad(coeffs):
    return _objective_and_grad(coeffs)[1]


def _random_initialization(rng):
    """Generate a random initial coefficient vector."""
    coeffs = rng.standard_normal(21)
    # Normalize to have reasonable magnitude
    coeffs /= np.linalg.norm(coeffs)
    coeffs *= 0.5
    return coeffs


def _polish(coeffs, max_iter=200, tol=1e-12):
    """High-precision polish using L-BFGS-B with tight tolerance."""
    try:
        result = minimize(
            _objective,
            coeffs,
            jac=_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50}
        )
        if result.success and np.all(np.isfinite(result.x)):
            return result.x
    except Exception:
        pass
    return coeffs


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 minimizing R = ||f||_4 / ||f||_2
    for degree-10 real spherical harmonics.
    
    Returns:
        coeffs: np.ndarray of shape (21,)
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    best_coeffs = np.ones(21, dtype=np.float64)
    best_score = _compute_ratio(best_coeffs)
    
    # Multiple restarts with different strategies
    rng = np.random.default_rng(42)
    
    # Strategy 1: Random restarts
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        coeffs = _random_initialization(rng)
        
        try:
            result = minimize(
                _objective,
                coeffs,
                jac=_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if np.all(np.isfinite(result.x)):
                score = _compute_ratio(result.x)
                if score < best_score:
                    best_score = score
                    best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Strategy 2: Perturb from best solution found
    if time.time() < deadline - 60.0:
        for i in range(10):
            if time.time() > deadline - 30.0:
                break
            
            perturbation = rng.standard_normal(21) * 0.1
            coeffs = best_coeffs + perturbation
            
            try:
                result = minimize(
                    _objective,
                    coeffs,
                    jac=_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if np.all(np.isfinite(result.x)):
                    score = _compute_ratio(result.x)
                    if score < best_score:
                        best_score = score
                        best_coeffs = result.x.copy()
            except Exception:
                continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 15.0:
        best_coeffs = _polish(best_coeffs, max_iter=200, tol=1e-14)
        final_score = _compute_ratio(best_coeffs)
        if final_score < best_score:
            best_score = final_score
    
    # Validate
    if not np.all(np.isfinite(best_coeffs)):
        return np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs *= (1.0 / max_abs)
    elif max_abs > 1e5:
        best_coeffs *= (1e5 / max_abs)
    
    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
