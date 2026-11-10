# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre, gammaln
from scipy.optimize import minimize


def _precompute():
    """Precompute quadrature nodes, weights, and spherical harmonic basis."""
    k = 10
    n_theta = 70
    x, w = np.polynomial.legendre.leggauss(n_theta)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    # Precompute P_10^m(cos(theta)) for m = 0..10
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = eval_legendre(10, m, x)

    # Normalization constants
    N = np.zeros(11)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + gammaln(11 - m) - gammaln(11 + m))
        N[m] = np.exp(log_N)

    # Build basis: B[m_idx, i, j] = Y_m(x_i, phi_j)
    B = np.zeros((21, n_theta, n_phi))
    for m_idx in range(21):
        m = m_idx - 10
        if m == 0:
            for j in range(n_phi):
                B[m_idx, :, j] = N[0] * P[0, :]
        elif m > 0:
            for j in range(n_phi):
                B[m_idx, :, j] = np.sqrt(2) * N[m] * P[m, :] * np.cos(m * phi[j])
        else:
            am = abs(m)
            for j in range(n_phi):
                B[m_idx, :, j] = np.sqrt(2) * N[am] * P[am, :] * np.sin(am * phi[j])

    # Quadrature weights
    W = np.outer(w, np.ones(n_phi)) * dphi

    return B, W, x, phi, n_theta, n_phi


def _compute_ratio_and_grad(c, B, W):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient w.r.t. c.
    f = sum_m c_m Y_m
    R^4 = ||f||_4^4 / ||f||_2^4 = (sum W f^4) / (sum W f^2)^2
    We minimize R, equivalently minimize log(R) or R itself.
    Let's minimize R directly.
    
    R = (sum W f^4)^{1/4} / (sum W f^2)^{1/2}
    
    Let A = sum W f^4, B2 = sum W f^2
    R = A^{1/4} / B2^{1/2}
    
    dR/dc_m = (1/4) A^{-3/4} (dA/dc_m) / B2^{1/2} - A^{1/4} * (1/2) B2^{-3/2} (dB2/dc_m)
    
    dA/dc_m = 4 sum W f^3 Y_m
    dB2/dc_m = 2 sum W f Y_m
    """
    f = np.tensordot(c, B, axes=(0, 0))  # shape (n_theta, n_phi)
    f2 = f * f
    f3 = f2 * f
    f4 = f2 * f2
    
    A = np.sum(W * f4)  # ||f||_4^4
    B2 = np.sum(W * f2)  # ||f||_2^2
    
    if B2 <= 0:
        return np.inf, np.zeros_like(c)
    
    R = (A ** 0.25) / np.sqrt(B2)
    
    # Gradient
    # dA/dc_m = 4 sum_{i,j} W_{i,j} f_{i,j}^3 B[m, i, j]
    dA_dc = 4.0 * np.tensordot(W * f3, B, axes=([0, 1], [1, 2]))
    # dB2/dc_m = 2 sum_{i,j} W_{i,j} f_{i,j} B[m, i, j]
    dB2_dc = 2.0 * np.tensordot(W * f, B, axes=([0, 1], [1, 2]))
    
    # dR/dc_m = (1/4) A^{-3/4} B2^{-1/2} dA_dc_m - (1/2) A^{1/4} B2^{-3/2} dB2_dc_m
    term1 = (0.25 * A**(-0.75)) / np.sqrt(B2) * dA_dc
    term2 = (0.5 * A**0.25) / (B2**1.5) * dB2_dc
    
    grad = term1 - term2
    
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimize the norm ratio ||f||_4 / ||f||_2 on the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill

    B, W, x, phi, n_theta, n_phi = _precompute()
    
    dim = 21
    
    # Precompute for speed: flatten B for tensordot
    B_flat = B.reshape(21, -1)  # (21, n_theta*n_phi)
    W_flat = W.reshape(-1)
    
    def compute_R(c):
        f = np.dot(c, B_flat)
        f2 = f * f
        f4 = f2 * f2
        A = np.dot(W_flat, f4)
        B2 = np.dot(W_flat, f2)
        if B2 <= 0:
            return np.inf
        return (A ** 0.25) / np.sqrt(B2)
    
    def compute_R_grad(c):
        f = np.dot(c, B_flat)
        f2 = f * f
        f3 = f2 * f
        f4 = f2 * f2
        A = np.dot(W_flat, f4)
        B2 = np.dot(W_flat, f2)
        if B2 <= 0:
            return np.inf, np.zeros(dim)
        R = (A ** 0.25) / np.sqrt(B2)
        
        # dA/dc = 4 * B_flat.T @ (W_flat * f3)
        dA_dc = 4.0 * B_flat.T @ (W_flat * f3)
        # dB2/dc = 2 * B_flat.T @ (W_flat * f)
        dB2_dc = 2.0 * B_flat.T @ (W_flat * f)
        
        term1 = (0.25 * A**(-0.75)) / np.sqrt(B2) * dA_dc
        term2 = (0.5 * A**0.25) / (B2**1.5) * dB2_dc
        grad = term1 - term2
        
        return R, grad
    
    best_coeffs = None
    best_R = np.inf
    
    def local_opt(c_init, max_iters=2000):
        """Optimize using L-BFGS-B with analytic gradient."""
        c = c_init.copy()
        c = c / np.linalg.norm(c)
        
        def obj(c_vec):
            R, _ = compute_R_grad(c_vec)
            return R
        
        def grad(c_vec):
            _, g = compute_R_grad(c_vec)
            return g
        
        result = minimize(
            obj,
            c,
            jac=grad,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
        )
        
        c_opt = result.x
        R_opt = compute_R(c_opt)
        
        return c_opt, R_opt
    
    # Generate diverse initializations
    np.random.seed(12345)
    
    initializations = []
    
    # 1. All ones
    initializations.append(np.ones(dim))
    
    # 2. Only m=0
    c0 = np.zeros(dim)
    c0[10] = 1.0
    initializations.append(c0.copy())
    
    # 3. Random unit vectors
    for trial in range(60):
        c = np.random.randn(dim)
        c /= np.linalg.norm(c)
        initializations.append(c)
    
    # 4. Alternating signs
    c_alt = np.ones(dim)
    c_alt[::2] = -1.0
    initializations.append(c_alt / np.linalg.norm(c_alt))
    
    # 5. Reduce m=0 weight
    c_low_m0 = np.ones(dim)
    c_low_m0[10] = 0.1
    initializations.append(c_low_m0 / np.linalg.norm(c_low_m0))
    
    # 6. Pure high-m modes
    for m_val in [5, 8, 10]:
        for sign in [1, -1]:
            c = np.zeros(dim)
            idx_pos = m_val + 10
            idx_neg = -m_val + 10
            c[idx_pos] = sign
            c[idx_neg] = sign
            initializations.append(c / np.linalg.norm(c))
    
    # 7. Equal weights
    c_equal = np.ones(dim)
    initializations.append(c_equal / np.linalg.norm(c_equal))
    
    # 8. Weighted by 1/sqrt(|m|+1)
    c_weighted = np.array([1.0/np.sqrt(abs(m_idx-10)+1) for m_idx in range(dim)])
    initializations.append(c_weighted / np.linalg.norm(c_weighted))
    
    # 9. Cosine-weighted
    c_cos = np.array([np.cos((m_idx-10)*np.pi/20) for m_idx in range(dim)])
    initializations.append(c_cos / np.linalg.norm(c_cos))
    
    # 10. Sine-weighted
    c_sin = np.array([np.sin((m_idx-10)*np.pi/20) if m_idx != 10 else 0 for m_idx in range(dim)])
    if np.linalg.norm(c_sin) > 1e-10:
        initializations.append(c_sin / np.linalg.norm(c_sin))
    
    # 11. Phase patterns
    for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        c_phase = np.array([np.cos((m_idx-10)*phase) for m_idx in range(dim)])
        initializations.append(c_phase / np.linalg.norm(c_phase))
        
        c_phase2 = np.array([np.sin((m_idx-10)*phase) for m_idx in range(dim)])
        if np.linalg.norm(c_phase2) > 1e-10:
            initializations.append(c_phase2 / np.linalg.norm(c_phase2))
    
    # 12. More random starts with different distributions
    for trial in range(30):
        c = np.random.exponential(1, size=dim)
        c /= np.linalg.norm(c)
        initializations.append(c)
        
        c = np.random.laplace(0, 1, size=dim)
        c /= np.linalg.norm(c)
        initializations.append(c)
    
    # Run local optimization from each initialization
    for i, c_init in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        c_opt, R = local_opt(c_init, max_iters=1000)
        
        if R < best_R:
            best_R = R
            best_coeffs = c_opt.copy()
    
    # Additional refinement with more random starts if time permits
    if time.time() < deadline - 40:
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            c = np.random.randn(dim)
            c /= np.linalg.norm(c)
            c_opt, R = local_opt(c, max_iters=500)
            if R < best_R:
                best_R = R
                best_coeffs = c_opt.copy()
    
    # Final refinement with more iterations
    if best_coeffs is not None and time.time() < deadline - 15:
        c_final, R_final = local_opt(best_coeffs, max_iters=3000)
        if R_final < best_R:
            best_R = R_final
            best_coeffs = c_final
    
    if best_coeffs is None:
        best_coeffs = np.ones(dim)
    
    # Ensure validity: max|c| in [1e-3, 1e5]
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (1e-3 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e5 / max_abs)
    
    # Remove any NaN or inf
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(dim)
    
    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
