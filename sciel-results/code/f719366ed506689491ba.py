# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.integrate import gauss_legendre
from scipy.optimize import minimize

def _build_quadrature(n_theta=70, n_phi=130):
    """Build quadrature weights and points for the unit sphere."""
    # Gauss-Legendre on [-1, 1] for cos(theta)
    x, w = gauss_legendre(n_theta)
    # x = cos(theta), w = weights for integral over cos(theta)
    # theta = arccos(x), sin(theta) = sqrt(1 - x^2)
    # phi points
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Meshgrid
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    sin_theta = np.sqrt(np.maximum(0, 1 - cos_theta**2))
    
    # Surface element dOmega = sin(theta) dtheta dphi
    # dtheta = -dx / sin(theta), so sin(theta) dtheta = -dx
    # Integral over sphere = integral_{-1}^{1} integral_0^{2pi} f dphi dx
    # So weight for each point is w_i * (2*pi/n_phi)
    weight = w[:, np.newaxis] * (2 * np.pi / n_phi)
    weight = weight * np.ones((1, n_phi))
    
    return cos_theta, phi_grid, weight


def _real_harmonics(cos_theta, phi, degree=10):
    """
    Compute real orthonormal spherical harmonics Y_m for m = -degree..degree.
    
    Y_0 = N_0 P_degree^0(cos theta)
    Y_m = sqrt(2) N_m P_degree^m(cos theta) cos(m phi) for m > 0
    Y_m = sqrt(2) N_|m| P_degree^|m|(cos theta) sin(|m| phi) for m < 0
    
    where N_m = sqrt((2*degree+1)/(4*pi) * (degree-m)!/(degree+m)!)
    """
    k = degree
    n = 2 * k + 1
    Y = np.zeros((n,) + cos_theta.shape, dtype=np.float64)
    
    # Precompute factorials for normalization
    # N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
    # We need (k-m)!/(k+m)! for m = 0..k
    # Use log gamma for numerical stability
    from scipy.special import gammaln
    
    for m in range(k + 1):
        # Normalization constant
        log_N = 0.5 * (np.log((2 * k + 1) / (4 * np.pi)) + gammaln(k - m + 1) - gammaln(k + m + 1))
        N_m = np.exp(log_N)
        
        # Associated Legendre polynomial with Condon-Shortley phase
        # lpmv(m, k, cos_theta) includes the (-1)^m factor
        P_m = lpmv(m, k, cos_theta)
        
        if m == 0:
            Y[k] = N_m * P_m  # Y_0 corresponds to index k (m=0)
        else:
            # For m > 0: Y_m = sqrt(2) * N_m * P_m * cos(m*phi)
            idx_pos = k + m
            Y[idx_pos] = np.sqrt(2) * N_m * P_m * np.cos(m * phi)
            
            # For m < 0: Y_{-m} = sqrt(2) * N_m * P_m * sin(m*phi)
            idx_neg = k - m
            Y[idx_neg] = np.sqrt(2) * N_m * P_m * np.sin(m * phi)
    
    return Y


def _compute_ratio(coeffs, cos_theta, phi, weight):
    """Compute ||f||_4 / ||f||_2 for given coefficients."""
    # Evaluate f at all quadrature points
    Y = _real_harmonics(cos_theta, phi, degree=10)
    # Y has shape (21, n_theta, n_phi)
    # coeffs has shape (21,)
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    
    # f has shape (n_theta, n_phi)
    f2 = f ** 2
    f4 = f ** 4
    
    # Integrate using quadrature weights
    norm2_sq = np.sum(f2 * weight)  # = ||f||_2^2
    norm4_sq = np.sum(f4 * weight)  # = ||f||_4^2
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(norm4_sq)
    
    return norm4 / norm2


def _objective(coeffs):
    """Objective function to minimize."""
    return _compute_ratio(coeffs, _COS_THETA, _PHI, _WEIGHT)


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 that minimize ||f||_4 / ||f||_2.
    """
    global _COS_THETA, _PHI, _WEIGHT
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Build quadrature
    _COS_THETA, _PHI, _WEIGHT = _build_quadrature(n_theta=70, n_phi=130)
    
    k = 10
    n = 2 * k + 1
    
    # Best solution found so far
    best_coeffs = np.ones(n, dtype=np.float64)
    best_ratio = _compute_ratio(best_coeffs, _COS_THETA, _PHI, _WEIGHT)
    
    # Generate multiple starting points
    np.random.seed(42)
    
    start_points = []
    # All ones
    start_points.append(np.ones(n, dtype=np.float64))
    
    # Random starts
    for i in range(10):
        start_points.append(np.random.randn(n))
    
    # Some structured starts
    # Only m=0
    c0 = np.zeros(n)
    c0[k] = 1.0
    start_points.append(c0)
    
    # Only m=±1
    c1 = np.zeros(n)
    c1[k-1] = 1.0
    c1[k+1] = 1.0
    start_points.append(c1)
    
    # Only m=±k
    ck = np.zeros(n)
    ck[0] = 1.0
    ck[-1] = 1.0
    start_points.append(ck)
    
    # Try to minimize from each start point
    for i, x0 in enumerate(start_points):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.success or result.fun < best_ratio:
                # Normalize to avoid issues
                c = result.x
                max_c = np.max(np.abs(c))
                if max_c > 1e-3 and max_c < 1e5:
                    ratio = result.fun
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_coeffs = c.copy()
        except Exception:
            pass
    
    # Final polish with trust-constr if time permits
    if time.time() < deadline - 60:
        try:
            # Use the best found so far as starting point
            result = minimize(
                _objective,
                best_coeffs,
                method='trust-constr',
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            
            if result.fun < best_ratio:
                c = result.x
                max_c = np.max(np.abs(c))
                if max_c > 1e-3 and max_c < 1e5:
                    best_ratio = result.fun
                    best_coeffs = c.copy()
        except Exception:
            pass
    
    # Validate and return
    if not np.all(np.isfinite(best_coeffs)):
        return np.ones(n, dtype=np.float64)
    
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        # Rescale if needed
        if max_c < 1e-3:
            best_coeffs *= (1.0 / max_c)
        elif max_c > 1e5:
            best_coeffs *= (1e5 / max_c)
    
    return best_coeffs


# EVOLVE-BLOCK-END
