# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import differential_evolution, minimize

def _precompute_basis():
    """
    Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
    on a 70x130 quadrature grid.
    Returns:
        B: Matrix of shape (n_pts, 21) where B[i, j] = Y_j(theta_i, phi_i)
        weights: Array of shape (n_pts,) with quadrature weights
    """
    k = 10
    n_theta = 70
    n_phi = 130
    dim = 2 * k + 1
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Compute P_10^m(x) for m=0..10
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
        
    # Normalization constants N_m
    # N_m = sqrt((2k+1)/(4pi) * (k-m)!/(k+m)!)
    N = np.zeros(11)
    for m in range(11):
        # Use log-gamma for numerical stability if needed, but factorials are fine for k=10
        # (10-m)! / (10+m)!
        # We can compute this iteratively to avoid large numbers
        num = 1.0
        den = 1.0
        for i in range(1, k - m + 1):
            num *= i
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    # Build B matrix (n_pts x dim)
    # n_pts = n_theta * n_phi
    B = np.zeros((n_theta * n_phi, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    
    sqrt2 = np.sqrt(2.0)
    
    # m = 0
    # Y_0 = N_0 P_10^0(x)
    val_theta = N[0] * P[0, :]
    B[:, 10] = np.repeat(val_theta, n_phi)
    
    for m in range(1, 11):
        idx_pos = m + 10
        idx_neg = -m + 10
        
        val_theta = sqrt2 * N[m] * P[m, :]
        
        cos_term = np.cos(m * phi)
        sin_term = np.sin(m * phi)
        
        B[:, idx_pos] = (val_theta[:, None] * cos_term[None, :]).ravel()
        B[:, idx_neg] = (val_theta[:, None] * sin_term[None, :]).ravel()
    
    return B, weights

def _compute_ratio(B, weights, c):
    """
    Compute R = ||f||_4 / ||f||_2
    """
    f = B @ c
    norm2_sq = np.dot(weights, f**2)
    if norm2_sq < 1e-15:
        return np.inf
    norm4_4 = np.dot(weights, f**4)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2

def _compute_grad(B, weights, c):
    """
    Compute gradient of R with respect to c.
    R = A^{1/4} B^{-1/2} where A = ||f||_4^4, B = ||f||_2^2
    """
    f = B @ c
    A = np.dot(weights, f**4)
    B_val = np.dot(weights, f**2)
    
    if B_val < 1e-15:
        return np.zeros_like(c)
    
    # dA/dc = 4 * B^T (f^3 * weights)
    # dB/dc = 2 * B^T (f * weights)
    f3w = f**3 * weights
    fw = f * weights
    
    grad_A = 4.0 * (B.T @ f3w)
    grad_B = 2.0 * (B.T @ fw)
    
    term1 = (A ** (-0.75)) * (B_val ** (-0.5)) * grad_A
    term2 = -0.5 * (A ** 0.25) * (B_val ** (-1.5)) * grad_B
    
    return term1 + term2

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    B, weights = _precompute_basis()
    
    # Initial guess: all ones
    best_coeffs = np.ones(21, dtype=np.float64)
    best_R = _compute_ratio(B, weights, best_coeffs)
    
    # Differential Evolution
    # Bounds: coefficients can be anything, but let's keep them reasonable.
    # The task says max|c| > 1e5 is invalid. So bounds should be within (-1e5, 1e5).
    # However, the optimal solution likely has small coefficients.
    # Let's use [-10, 10] for DE to focus search, then polish.
    bounds = [(-10.0, 10.0)] * 21
    
    def objective(c):
        return _compute_ratio(B, weights, c)
    
    # Run DE
    # Use a smaller population to allow more iterations within time limit
    # popsize=15 is default, let's try 10 or 12
    # maxiter: estimate based on time
    # Each evaluation is B @ c (21x9100) + dot products. ~100k flops. Very fast.
    # 9100 points. 21 coeffs.
    # Let's allow a generous maxiter.
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=15,
            seed=42,
            tol=1e-8,
            polish=False  # We will polish manually with gradient
        )
        if result.fun < best_R:
            best_coeffs = result.x
            best_R = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        if not np.all(np.isfinite(best_coeffs)):
            best_coeffs = np.ones(21)
        return best_coeffs
    
    # Local polish from the best DE result using gradient
    try:
        res = minimize(
            objective,
            best_coeffs,
            method='L-BFGS-B',
            jac=lambda c: _compute_grad(B, weights, c),
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < best_R:
            best_coeffs = res.x
            best_R = res.fun
    except Exception:
        pass
    
    # If we have time left, try a few random restarts with local polish
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        # Generate some random starting points
        n_restarts = min(5, int(remaining_time / 10))
        for _ in range(n_restarts):
            if time.time() > deadline:
                break
            c0 = np.random.normal(0, 1, 21)
            try:
                res = minimize(
                    objective,
                    c0,
                    method='L-BFGS-B',
                    jac=lambda c: _compute_grad(B, weights, c),
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                if res.fun < best_R:
                    best_coeffs = res.x
                    best_R = res.fun
            except Exception:
                pass
    
    # Final validation
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
    
    # Ensure max|c| is within valid range
    max_c = np.max(np.abs(best_coeffs))
    if max_c > 1e5:
        best_coeffs = best_coeffs / max_c * 1e5
    if max_c < 1e-3 and max_c > 0:
        # If too small, scale up? No, if it's a valid local min, leave it.
        # But if it's exactly 0, it's invalid.
        if max_c == 0:
            best_coeffs = np.ones(21)
    
    return best_coeffs

# EVOLVE-BLOCK-END
