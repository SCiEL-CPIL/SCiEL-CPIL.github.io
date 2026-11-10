# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time


def _build_quadrature(n_theta=70, n_phi=130):
    """
    Build 70-point Gauss-Legendre quadrature in cos(theta) times 130 equally
    spaced phi points. Returns arrays for cos_theta, sin_theta, phi, and weights.
    The total surface measure integrates to 4*pi.
    """
    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x = cos(theta), theta = arccos(x)
    cos_theta = x
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - x**2))
    theta = np.arccos(np.clip(x, -1.0, 1.0))
    
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    
    # Surface element: dOmega = sin(theta) dtheta dphi
    # But we integrate over x = cos(theta), so dtheta = -dx/sin(theta)
    # Thus sin(theta) dtheta = -dx, and the integral becomes:
    # ∫∫ f sin(theta) dtheta dphi = ∫∫ f dx dphi
    # So the weight in x is just w_i, and we sum over phi with weight 2*pi/n_phi
    # Total weight: sum_i w_i * (2*pi/n_phi) = 2 * 2*pi/n_phi = 4*pi/n_phi * n_phi/2... 
    # Actually: sum_i w_i = 2 (for GL on [-1,1]), so total = 2 * 2*pi/n_phi * n_phi = 4*pi. Good.
    
    # Build full grid
    cos_theta_grid, phi_grid = np.meshgrid(cos_theta, phi, indexing='ij')
    # Weights: w_i * (2*pi/n_phi)
    w_theta = w * (2.0 * np.pi / n_phi)
    weights = np.tile(w_theta[:, None], (1, n_phi))
    
    return cos_theta_grid, sin_theta_grid, phi_grid, weights


def _assoc_legendre(n, m, x):
    """
    Compute associated Legendre function P_n^m(x) with Condon-Shortley phase.
    Uses scipy.special.lpmv which includes the Condon-Shortley phase (-1)^m.
    """
    from scipy.special import lpmv
    return lpmv(m, n, x)


def _build_Y_matrix(cos_theta, sin_theta, phi, k=10):
    """
    Build the matrix of real orthonormal spherical harmonics of degree k.
    Columns correspond to m = -k, ..., 0, ..., k.
    Y_0 = N_0 P_k^0(cos theta)
    Y_m = sqrt(2) N_m P_k^m(cos theta) cos(m phi) for m > 0
    Y_m = sqrt(2) N_|m| P_k^|m|(cos theta) sin(|m| phi) for m < 0
    N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
    """
    n_pts = cos_theta.shape[0]
    Y = np.zeros((n_pts, 2 * k + 1))
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        # Normalization
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * factorial(k - abs_m) / factorial(k + abs_m))
        P = _assoc_legendre(k, abs_m, cos_theta)
        
        if m == 0:
            Y[:, m_idx] = N_m * P
        elif m > 0:
            Y[:, m_idx] = np.sqrt(2.0) * N_m * P * np.cos(m * phi)
        else:  # m < 0
            Y[:, m_idx] = np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi)
    
    return Y


def _compute_ratio(coeffs, Y, weights):
    """
    Compute R = ||f||_4 / ||f||_2 where f = Y @ coeffs.
    """
    f = Y @ coeffs
    f2 = f * f
    f4 = f2 * f2
    
    # Integrate with weights
    l2_sq = np.sum(weights * f2)
    l4 = np.sum(weights * f4)
    
    if l2_sq <= 0:
        return np.inf
    
    l2 = np.sqrt(l2_sq)
    l4_norm = np.sqrt(l4)  # Wait, ||f||_4 = (∫|f|^4)^(1/4)
    l4_norm = l4 ** 0.25
    
    return l4_norm / l2


def _objective(coeffs, Y, weights):
    """
    Objective function to minimize: R = ||f||_4 / ||f||_2.
    We minimize this directly.
    """
    return _compute_ratio(coeffs, Y, weights)


def _gradient(coeffs, Y, weights):
    """
    Compute gradient of R = ||f||_4 / ||f||_2 with respect to coeffs.
    Let f = Y @ coeffs.
    L2 = (∫ f^2)^(1/2), L4 = (∫ f^4)^(1/4)
    R = L4 / L2
    
    dR/dc_j = (1/L2) dL4/dc_j - (L4/L2^2) dL2/dc_j
    
    dL2/dc_j = (1/(2 L2)) * 2 ∫ f * Y_j dΩ = (1/L2) ∫ f Y_j
    dL4/dc_j = (1/(4 L4^3)) * 4 ∫ f^3 Y_j dΩ = (1/L4^3) ∫ f^3 Y_j
    
    So: dR/dc_j = (1/L2)(1/L4^3) ∫ f^3 Y_j - (L4/L2^2)(1/L2) ∫ f Y_j
               = (1/(L2 L4^3)) ∫ f^3 Y_j - (L4/L2^3) ∫ f Y_j
    """
    f = Y @ coeffs
    f2 = f * f
    f3 = f * f2
    f4 = f2 * f2
    
    l2_sq = np.sum(weights * f2)
    l4 = np.sum(weights * f4)
    
    if l2_sq <= 0 or l4 <= 0:
        return np.zeros_like(coeffs)
    
    l2 = np.sqrt(l2_sq)
    l4_norm = l4 ** 0.25
    l4_cubed = l4_norm ** 3
    
    # ∫ f^3 Y_j and ∫ f Y_j for each j
    int_f3_Y = Y.T @ (weights * f3)
    int_f_Y = Y.T @ (weights * f)
    
    grad = (1.0 / (l2 * l4_cubed)) * int_f3_Y - (l4_norm / (l2 ** 3)) * int_f_Y
    
    return grad


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimizes R = ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s kill
    
    # Build quadrature
    cos_theta, sin_theta, phi, weights = _build_quadrature(n_theta=70, n_phi=130)
    Y = _build_Y_matrix(cos_theta, sin_theta, phi, k=10)
    
    # Precompute Y.T @ weights for efficiency
    YTW = Y.T * weights.T  # shape (21, n_pts)
    
    def _fast_ratio(coeffs):
        f = Y @ coeffs
        f2 = f * f
        f4 = f2 * f2
        l2_sq = np.sum(weights * f2)
        l4 = np.sum(weights * f4)
        if l2_sq <= 0 or l4 <= 0:
            return np.inf
        return (l4 ** 0.25) / np.sqrt(l2_sq)
    
    def _fast_grad(coeffs):
        f = Y @ coeffs
        f2 = f * f
        f3 = f * f2
        f4 = f2 * f2
        l2_sq = np.sum(weights * f2)
        l4 = np.sum(weights * f4)
        if l2_sq <= 0 or l4 <= 0:
            return np.zeros(21)
        l2 = np.sqrt(l2_sq)
        l4_norm = l4 ** 0.25
        l4_cubed = l4_norm ** 3
        int_f3_Y = Y.T @ (weights * f3)
        int_f_Y = Y.T @ (weights * f)
        return (1.0 / (l2 * l4_cubed)) * int_f3_Y - (l4_norm / (l2 ** 3)) * int_f_Y
    
    # Start with all-ones
    coeffs0 = np.ones(21, dtype=np.float64)
    
    best_coeffs = coeffs0.copy()
    best_ratio = _fast_ratio(best_coeffs)
    
    # Try multiple starting points
    start_points = [coeffs0]
    
    # Add some structured starting points
    for trial in range(10):
        if time.time() > deadline - 60.0:
            break
        rng = np.random.RandomState(trial * 12345 + 67890)
        c = rng.randn(21)
        c /= np.linalg.norm(c)
        start_points.append(c)
    
    # Optimize from each starting point
    for i, c0 in enumerate(start_points):
        if time.time() > deadline - 60.0:
            break
        
        try:
            result = minimize(
                _fast_ratio,
                c0,
                jac=_fast_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x.copy()
                
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 30.0:
        try:
            result = minimize(
                _fast_ratio,
                best_coeffs,
                jac=_fast_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Validate
    if not np.all(np.isfinite(best_coeffs)):
        return np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs *= 1.0
    if max_abs > 1e5:
        best_coeffs *= 1e5 / max_abs
    
    return best_coeffs


# EVOLVE-BLOCK-END
