# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import factorial, lpmv
import time


def _build_quadrature_and_harmonics(k=10, n_theta=70, n_phi=130):
    """
    Build the quadrature grid and precompute the real orthonormal spherical
    harmonic values at all grid points.
    
    Returns:
        f_values: array of shape (n_theta, n_phi, 2k+1) where f_values[:, :, m+k]
                  is Y_k^m evaluated at (theta_i, phi_j).
        weights: array of shape (n_theta, n_phi) with quadrature weights.
    """
    # Gauss-Legendre quadrature in cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x is in [-1, 1], this is cos(theta)
    # theta = arccos(x), but we work with cos(theta) directly
    # The surface element is dOmega = dphi dcos(theta), so the weight for
    # integrating over the sphere is w_i * dphi
    
    cos_theta = x  # shape (n_theta,)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)  # shape (n_phi,)
    dphi = 2 * np.pi / n_phi
    
    # Build the harmonic values
    # For each m in -k..k, compute Y_k^m at each (theta, phi)
    # Y_0 = N_0 * P_k^0(cos_theta)
    # Y_m = sqrt(2) * N_m * P_k^m(cos_theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_k^|m|(cos_theta) * sin(|m|*phi) for m < 0
    # N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
    
    n_m = 2 * k + 1
    f_values = np.zeros((n_theta, n_phi, n_m), dtype=np.float64)
    
    # Precompute associated Legendre polynomials P_k^m(cos_theta) for m=0..k
    # Using scipy.special.lpmv which includes Condon-Shortley phase
    P = np.zeros((k + 1, n_theta), dtype=np.float64)
    for m in range(k + 1):
        P[m, :] = lpmv(m, k, cos_theta)
    
    # Compute normalization constants
    N = np.zeros(k + 1, dtype=np.float64)
    for m in range(k + 1):
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * factorial(k - m) / factorial(k + m))
    
    # Build f_values
    for m in range(-k, k + 1):
        idx = m + k
        if m == 0:
            f_values[:, :, idx] = N[0] * P[0, :]
        elif m > 0:
            f_values[:, :, idx] = np.sqrt(2) * N[m] * P[m, :][:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            abs_m = -m
            f_values[:, :, idx] = np.sqrt(2) * N[abs_m] * P[abs_m, :][:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    # Weights: w_i * dphi for each (i, j)
    weights = np.outer(w, np.full(n_phi, dphi))
    
    return f_values, weights


def _compute_ratio(coeffs, f_values, weights):
    """
    Compute ||f||_4 / ||f||_2 where f = sum_m c_m Y_m.
    
    Args:
        coeffs: array of shape (21,)
        f_values: precomputed harmonic values, shape (n_theta, n_phi, 21)
        weights: quadrature weights, shape (n_theta, n_phi)
    
    Returns:
        ratio: ||f||_4 / ||f||_2
    """
    # f at each grid point: shape (n_theta, n_phi)
    f = np.tensordot(coeffs, f_values, axes=([0], [2]))
    
    # ||f||_2^2 = sum f^2 * weights
    norm2_sq = np.sum(f ** 2 * weights)
    if norm2_sq <= 0:
        return np.inf
    
    # ||f||_4^4 = sum f^4 * weights
    norm4_4 = np.sum(f ** 4 * weights)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _compute_gradient(coeffs, f_values, weights):
    """
    Compute gradient of R = ||f||_4 / ||f||_2 with respect to coeffs.
    
    Let f = Y^T c (vector of f values at grid points)
    R = (sum f^4 w)^(1/4) / (sum f^2 w)^(1/2)
    
    Using log differentiation:
    log R = (1/4) log(sum f^4 w) - (1/2) log(sum f^2 w)
    
    d(log R)/dc = (1/4) * (4 f^3 w / sum f^4 w) * df/dc - (1/2) * (2 f w / sum f^2 w) * df/dc
               = (f^3 w / sum f^4 w) * df/dc - (f w / sum f^2 w) * df/dc
    
    df/dc_j = Y_j (the j-th harmonic)
    
    dR/dc = R * d(log R)/dc
    """
    f = np.tensordot(coeffs, f_values, axes=([0], [2]))  # shape (n_theta, n_phi)
    
    norm2_sq = np.sum(f ** 2 * weights)
    norm4_4 = np.sum(f ** 4 * weights)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    R = norm4 / norm2
    
    # d(log R)/dc_j = (f^3 w / norm4_4) * Y_j - (f w / norm2_sq) * Y_j
    # = Y_j * (f^3 w / norm4_4 - f w / norm2_sq)
    
    # Compute the coefficient vector: for each grid point, the factor
    factor = (f ** 3 * weights) / norm4_4 - (f * weights) / norm2_sq  # shape (n_theta, n_phi)
    
    # gradient_j = R * sum_{i,j} factor[i,j] * f_values[i,j,j]
    # This is a matrix-vector product: gradient = R * (f_values.T @ factor.reshape(-1)).reshape(21)
    # More efficiently: gradient_j = R * sum over grid of factor * Y_j
    grad = np.tensordot(factor, f_values, axes=([0, 1], [0, 1]))  # shape (21,)
    grad *= R
    
    return grad


def _lbfgs_optimize(initial_coeffs, f_values, weights, max_iter=500, tol=1e-10):
    """
    Simple L-BFGS implementation for minimizing R(coeffs).
    """
    n = len(initial_coeffs)
    m = 10  # number of L-BFGS memory vectors
    
    x = initial_coeffs.copy()
    f_val = _compute_ratio(x, f_values, weights)
    g = _compute_gradient(x, f_values, weights)
    
    # L-BFGS state
    s_list = []
    y_list = []
    
    best_x = x.copy()
    best_f = f_val
    
    for it in range(max_iter):
        # Build two-loop recursion for L-BFGS direction
        if len(s_list) == 0:
            direction = -g
        else:
            # Two-loop recursion
            rho = [1.0 / np.dot(s_i, y_i) for s_i, y_i in zip(s_list, y_list)]
            
            # Backward loop
            q = g.copy()
            alpha = [0.0] * len(s_list)
            for i in range(len(s_list) - 1, -1, -1):
                alpha[i] = rho[i] * np.dot(s_list[i], q)
                q = q - alpha[i] * y_list[i]
            
            # Initial Hessian approximation
            if len(s_list) > 0:
                gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
            else:
                gamma = 1.0
            r = gamma * q
            
            # Forward loop
            for i in range(len(s_list)):
                beta = rho[i] * np.dot(y_list[i], r)
                r = r + (alpha[i] - beta) * s_list[i]
            
            direction = -r
        
        # Line search (backtracking)
        step = 1.0
        for ls in range(30):
            x_new = x + step * direction
            f_new = _compute_ratio(x_new, f_values, weights)
            
            if np.isnan(f_new) or np.isinf(f_new):
                step *= 0.5
                continue
            
            # Armijo condition
            if f_new <= f_val + 1e-4 * step * np.dot(g, direction):
                break
            step *= 0.5
        else:
            break
        
        g_new = _compute_gradient(x_new, f_values, weights)
        
        s = x_new - x
        y = g_new - g
        
        sy = np.dot(s, y)
        if sy > 1e-10:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = x_new
        f_val = f_new
        g = g_new
        
        if f_val < best_f:
            best_f = f_val
            best_x = x.copy()
        
        if np.linalg.norm(g) < tol:
            break
    
    return best_x, best_f


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 that minimize ||f||_4 / ||f||_2
    where f = sum_m c_m Y_10^m with real orthonormal spherical harmonics.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    k = 10
    n_coeffs = 2 * k + 1
    
    # Build quadrature and harmonics
    f_values, weights = _build_quadrature_and_harmonics(k, n_theta=70, n_phi=130)
    
    # Multiple random restarts
    best_coeffs = np.ones(n_coeffs, dtype=np.float64)
    best_ratio = _compute_ratio(best_coeffs, f_values, weights)
    
    np.random.seed(42)
    
    n_restarts = 50  # number of random restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:  # safety margin
            break
        
        # Random initial point
        if restart == 0:
            x0 = np.ones(n_coeffs, dtype=np.float64)
        else:
            x0 = np.random.randn(n_coeffs) * 0.5
            # Ensure non-degenerate
            if np.max(np.abs(x0)) < 1e-3:
                x0[0] = 1.0
        
        # Run L-BFGS
        x_opt, f_opt = _lbfgs_optimize(x0, f_values, weights, max_iter=300, tol=1e-12)
        
        # Validate
        if np.max(np.abs(x_opt)) < 1e-3 or np.max(np.abs(x_opt)) > 1e5:
            continue
        
        if f_opt < best_ratio:
            best_ratio = f_opt
            best_coeffs = x_opt.copy()
    
    # Final validation
    if np.max(np.abs(best_coeffs)) < 1e-3:
        best_coeffs = np.ones(n_coeffs, dtype=np.float64)
    
    return best_coeffs


# EVOLVE-BLOCK-END
