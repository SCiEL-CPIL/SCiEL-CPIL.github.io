# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time


def _build_basis(k=10):
    """
    Build real orthonormal spherical harmonic basis values on a fixed quadrature grid.
    Returns:
        Y: array of shape (2k+1, n_theta, n_phi)
        weights: array of shape (n_theta, n_phi) for surface integration
    """
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x is cos(theta)
    theta = np.arccos(x)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Surface measure: dOmega = sin(theta) dtheta dphi
    # Since x = cos(theta), sin(theta) dtheta = -dx, so weight factor is 1 (after abs)
    # Actually: integral f dOmega = integral_0^{2pi} integral_{-1}^{1} f(x,phi) dx dphi
    # The Gauss-Legendre weights w already account for the dx integration.
    # So the total weight for each (theta_i, phi_j) is w_i * (2pi/n_phi)
    # Wait, the problem says "70-point Gauss-Legendre quadrature in cos theta times 130 equally spaced phi".
    # The integral over phi is approximated by (2pi/n_phi) * sum_j f(x_i, phi_j).
    # The integral over x is sum_i w_i f(x_i, phi_j).
    # So weight[i,j] = w[i] * (2pi / n_phi)

    phi_weights = np.full(n_phi, 2 * np.pi / n_phi)
    weights = np.outer(w, phi_weights)

    # Compute spherical harmonics
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^{|m|}(cos theta) sin(|m| phi) for m < 0
    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)  [Note: 21 = 2k+1]
    
    coeffs = np.zeros((2 * k + 1, n_theta, n_phi))
    
    # Precompute factorials or use gamma
    # N_m = sqrt((2k+1)/(4*pi) * (k-m)! / (k+m)!)
    # We can compute this iteratively or using log-gamma for stability, but k=10 is small.
    from math import factorial, pi
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        # N_m
        num = factorial(k - abs_m)
        den = factorial(k + abs_m)
        N_m = np.sqrt((2 * k + 1) / (4 * pi) * num / den)
        
        # P_10^m(cos theta) with Condon-Shortley phase
        # scipy.special.lpmv(m, k, x) includes the Condon-Shortley phase (-1)^m
        P = lpmv(abs_m, k, x)  # shape (n_theta,)
        
        if m == 0:
            coeffs[k, :, :] = N_m * P[:, np.newaxis]
        elif m > 0:
            coeffs[k + m, :, :] = np.sqrt(2) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            # m < 0
            coeffs[k + m, :, :] = np.sqrt(2) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return coeffs, weights


def _compute_R(coeffs, Y, weights):
    """
    Compute R = ||f||_4 / ||f||_2
    """
    f = np.tensordot(coeffs, Y, axes=([0], [0]))  # shape (n_theta, n_phi)
    
    # ||f||_2^2 = sum f^2 * weights
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    
    # ||f||_4^4 = sum f^4 * weights
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    
    if norm2_sq <= 0:
        return np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2


def _evaluate_coefficients(c, Y, weights):
    """
    Evaluate the objective function (we want to minimize R, but the score is 1/R).
    Returns the score (1/R) for optimization purposes if we were maximizing,
    or R if minimizing. Let's minimize R.
    """
    return _compute_R(c, Y, weights)


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic that minimize ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin for 600s limit
    
    k = 10
    Y, weights = _build_basis(k)
    
    # Initial guess: all ones (baseline)
    best_coeffs = np.ones(2 * k + 1, dtype=np.float64)
    best_R = _compute_R(best_coeffs, Y, weights)
    
    # Optimization: Minimize R
    # We can use scipy.optimize.minimize
    from scipy.optimize import minimize
    
    # Use L-BFGS-B or BFGS
    # Since R is smooth, gradient-based methods should work well.
    # We need the gradient of R with respect to c.
    # R(c) = (sum (c^T Y)^4 w)^{1/4} / (sum (c^T Y)^2 w)^{1/2}
    # Let A = Y (flattened), w = weights (flattened)
    # f = A c
    # R = (w^T f^4)^{1/4} / (w^T f^2)^{1/2}
    
    # It might be easier to use numerical gradients or just let scipy handle it.
    # Given the small dimension (21), numerical gradient is fine.
    
    def objective(c):
        return _compute_R(c, Y, weights)
    
    # Try a few different initial points and optimizers
    # 1. All ones
    # 2. Only m=0
    # 3. Random
    
    # Let's use a multi-start approach with L-BFGS-B
    # Since the function is homogeneous of degree 1 (R(λc) = R(c)), we can fix ||c||_2 = 1
    # or just let the optimizer handle it.
    
    # Actually, R is invariant to scaling. So we can constrain ||c||_2 = 1.
    # But standard minimize doesn't do constraints easily without SLSQP.
    # L-BFGS-B is fast. Let's just minimize R directly.
    
    # To avoid scaling issues, we can normalize c inside the objective?
    # No, R is scale invariant, so the gradient might be zero if we don't constrain?
    # d/dc (N/D) = (D dN/dc - N dD/dc) / D^2
    # Since N and D are homogeneous of degree 1 and 2 respectively?
    # N = ||f||_4 ~ (sum (c^T Y)^4 w)^{1/4}. Degree 1 in c.
    # D = ||f||_2 ~ (sum (c^T Y)^2 w)^{1/2}. Degree 1 in c.
    # So R is degree 0.
    # The gradient of a homogeneous function of degree 0 is not zero generally.
    
    # Let's just run minimize.
    
    # Multi-start
    rng = np.random.default_rng(42)
    starts = [best_coeffs]
    
    # Add some structured starts
    for m in range(-k, k+1):
        c = np.zeros(2*k+1)
        c[k+m] = 1.0
        starts.append(c)
    
    # Add random starts
    for _ in range(10):
        c = rng.normal(size=2*k+1)
        starts.append(c)
    
    best_R = np.inf
    best_c = None
    
    for c0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, c0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.success or res.fun < best_R:
                if np.isfinite(res.fun) and res.fun < best_R:
                    best_R = res.fun
                    best_c = res.x
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline:
            break
            
    if best_c is None:
        return best_coeffs
        
    # Verify validity
    if len(best_c) != 21:
        return best_coeffs
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return best_coeffs
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Scale to valid range
        if max_abs < 1e-3:
            best_c = best_c * (1.0 / max_abs)
        else:
            best_c = best_c * (1.0 / max_abs)
            
    return best_c


# EVOLVE-BLOCK-END
