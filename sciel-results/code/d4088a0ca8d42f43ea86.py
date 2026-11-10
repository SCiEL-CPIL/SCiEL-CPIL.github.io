# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
import time


def _build_harmonics(k=10, n_theta=70, n_phi=130):
    """
    Build the real orthonormal spherical harmonics of degree k on a
    Gauss-Legendre theta grid and equally spaced phi grid.
    
    Returns:
        Y: array of shape (2k+1, n_theta, n_phi) where Y[m+k] is the
           real harmonic for m in -k..k.
        weights: array of shape (n_theta, n_phi) for the surface integral
                 weights (including the 1/sin(theta) Jacobian factor
                 and the GL weights).
    """
    # Gauss-Legendre nodes and weights in [-1, 1] for cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x = cos(theta), so theta = arccos(x)
    theta = np.arccos(x)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Build grid
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    sin_theta = np.sqrt(1 - x**2)
    sin_theta = np.maximum(sin_theta, 1e-15)
    
    # Surface element: dOmega = sin(theta) dtheta dphi
    # dtheta = -dx / sin(theta), so sin(theta) * dtheta = -dx
    # We integrate over x from 1 to -1, so sin(theta) dtheta = -dx
    # The integral becomes ∫∫ f(x, phi) dx dphi with x from -1 to 1
    # So the weight is just w_x * dphi
    dphi = 2 * np.pi / n_phi
    # weights[i, j] = w[i] * dphi
    weights = np.outer(w, np.ones(n_phi)) * dphi
    
    # Build the 2k+1 real orthonormal spherical harmonics
    Y = np.zeros((2 * k + 1, n_theta, n_phi), dtype=np.float64)
    
    for m in range(0, k + 1):
        # Normalization: N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * factorial(k - m) / factorial(k + m))
        
        # Associated Legendre function P_k^m(cos(theta)) with Condon-Shortley phase
        # lpmv(m, k, x) gives P_k^m(x) with the Condon-Shortley phase (-1)^m
        P_m = lpmv(m, k, x)  # shape (n_theta,)
        
        if m == 0:
            Y[k] = N_m * P_m[:, np.newaxis] * np.ones((1, n_phi))
        else:
            # Y_m = sqrt(2) * N_m * P^m * cos(m*phi)
            cos_m_phi = np.cos(m * phi)[np.newaxis, :]
            Y[k + m] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * cos_m_phi
            
            # Y_{-m} = sqrt(2) * N_m * P^m * sin(m*phi)
            sin_m_phi = np.sin(m * phi)[np.newaxis, :]
            Y[k - m] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * sin_m_phi
    
    return Y, weights


def _compute_ratio(coeffs, Y, weights):
    """
    Compute R = ||f||_4 / ||f||_2 for f = sum c_m Y_m.
    """
    # f = sum_m c_m Y_m
    f = np.tensordot(coeffs, Y, axes=([0], [0]))  # shape (n_theta, n_phi)
    
    # ||f||_2^2 = ∫ f^2 dOmega
    f2 = f ** 2
    norm2_sq = np.sum(f2 * weights)
    
    if norm2_sq <= 0:
        return np.inf
    
    # ||f||_4^4 = ∫ f^4 dOmega
    f4 = f ** 4
    norm4_4 = np.sum(f4 * weights)
    
    if norm4_4 <= 0:
        return 0.0
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _objective(coeffs, Y, weights):
    """Objective function for optimization."""
    return _compute_ratio(coeffs, Y, weights)


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 that minimize the L4/L2
    ratio of the degree-10 spherical harmonic f = sum c_m Y_m.
    """
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s limit
    
    k = 10
    Y, weights = _build_harmonics(k=k, n_theta=70, n_phi=130)
    
    best_coeffs = np.ones(2 * k + 1, dtype=np.float64)
    best_ratio = _compute_ratio(best_coeffs, Y, weights)
    
    # Try several initial guesses
    initial_guesses = []
    
    # All ones
    initial_guesses.append(np.ones(2 * k + 1))
    
    # Only m=0
    c0 = np.zeros(2 * k + 1)
    c0[k] = 1.0
    initial_guesses.append(c0)
    
    # Only m=+10
    c10 = np.zeros(2 * k + 1)
    c10[2 * k] = 1.0
    initial_guesses.append(c10)
    
    # Only m=-10
    cm10 = np.zeros(2 * k + 1)
    cm10[0] = 1.0
    initial_guesses.append(cm10)
    
    # Alternating signs
    alt = np.ones(2 * k + 1)
    alt[::2] = -1.0
    initial_guesses.append(alt)
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        initial_guesses.append(rng.standard_normal(2 * k + 1))
    
    # Local optimization from each initial guess
    for i0, x0 in enumerate(initial_guesses):
        if time.time() - start_time > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(Y, weights),
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_ratio:
                ratio = result.fun
                if np.isfinite(ratio) and ratio < best_ratio:
                    best_ratio = ratio
                    best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Differential evolution for global search
    if time.time() - start_time < deadline - 60:
        bounds = [(-10, 10)] * (2 * k + 1)
        
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                args=(Y, weights),
                maxiter=100,
                popsize=15,
                tol=1e-8,
                seed=123,
                polish=False
            )
            
            if de_result.fun < best_ratio:
                best_ratio = de_result.fun
                best_coeffs = de_result.x.copy()
                
                # Polish with local optimization
                if time.time() - start_time < deadline - 10:
                    try:
                        polish_result = minimize(
                            _objective,
                            best_coeffs,
                            args=(Y, weights),
                            method='BFGS',
                            options={'maxiter': 1000, 'gtol': 1e-12}
                        )
                        if polish_result.fun < best_ratio:
                            best_ratio = polish_result.fun
                            best_coeffs = polish_result.x.copy()
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Final polish
    if time.time() - start_time < deadline - 5:
        try:
            final_result = minimize(
                _objective,
                best_coeffs,
                args=(Y, weights),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-14}
            )
            if final_result.fun < best_ratio:
                best_ratio = final_result.fun
                best_coeffs = final_result.x.copy()
        except Exception:
            pass
    
    # Validate
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if best_coeffs.shape != (2 * k + 1,):
        best_coeffs = np.ones(2 * k + 1, dtype=np.float64)
    
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(2 * k + 1, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs / max_abs * 1e-2 if max_abs > 0 else np.ones(2 * k + 1)
    if max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e4
    
    return best_coeffs


# EVOLVE-BLOCK-END
