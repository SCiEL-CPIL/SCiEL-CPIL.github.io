# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre
from scipy.optimize import minimize


def _precompute():
    """Precompute quadrature nodes, weights, and spherical harmonic basis."""
    k = 10
    
    # Gauss-Legendre quadrature for cos(theta) in [-1, 1]
    # 70-point GL quadrature is exact for polynomials of degree up to 139
    # Our integrands involve (P_10^m)^4 which is degree 40 in cos(theta), so 70 is plenty
    n_theta = 70
    x, w = np.polynomial.legendre.leggauss(n_theta)  # x in [-1, 1], w are weights
    # x = cos(theta), dOmega = sin(theta) dtheta dphi = d(cos(theta)) dphi
    # So integral over sphere = sum_i sum_j w_i f(x_i, phi_j)
    
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Build the 21 real orthonormal spherical harmonics Y_m for m = -10..10
    # Y_0 = N_0 P_10^0
    # Y_m = sqrt(2) N_m P_10^m cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m| sin(|m| phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    
    # Precompute P_10^m(cos(theta)) for m = 0..10 at all quadrature nodes
    # Using scipy's eval_legendre which includes Condon-Shortley phase
    P = np.zeros((11, n_theta))  # P[m, i] = P_10^m(x_i)
    for m in range(11):
        P[m, :] = eval_legendre(10, m, x)
    
    # Normalization constants
    N = np.zeros(11)
    for m in range(11):
        # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        # Use log-gamma for numerical stability
        from scipy.special import gammaln
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + gammaln(11 - m) - gammaln(11 + m))
        N[m] = np.exp(log_N)
    
    # Build basis matrix: B[m_idx, i, j] where m_idx=0..20 corresponds to m=-10..10
    # B[m_idx, i, j] = Y_m(x_i, phi_j)
    # Shape: (21, n_theta, n_phi)
    B = np.zeros((21, n_theta, n_phi))
    
    for m_idx in range(21):
        m = m_idx - 10  # actual m value
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
    
    # Quadrature weights: for each (i, j), weight = w_i * dphi
    # Total integral approximation: sum_{i,j} w_i * dphi * f(x_i, phi_j)
    W = np.outer(w, np.ones(n_phi)) * dphi  # shape (n_theta, n_phi)
    
    return B, W, x, phi, n_theta, n_phi


def _evaluate_ratio(coeffs, B, W):
    """Compute ||f||_4 / ||f||_2 using precomputed basis and weights."""
    # f = sum_m c_m Y_m
    # f[i, j] = sum_m c_m B[m, i, j]
    f = np.tensordot(coeffs, B, axes=(0, 0))  # shape (n_theta, n_phi)
    
    # ||f||_2^2 = sum_{i,j} W[i,j] * f[i,j]^2
    f2 = f * f
    norm2_sq = np.sum(W * f2)
    
    # ||f||_4^4 = sum_{i,j} W[i,j] * f[i,j]^4
    f4 = f2 * f2
    norm4_4 = np.sum(W * f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimize the norm ratio ||f||_4 / ||f||_2 on the unit sphere.
    
    Returns:
        coeffs: np.ndarray of shape (21,) with real coefficients c_m,
            ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    B, W, x, phi, n_theta, n_phi = _precompute()
    
    k = 10
    dim = 2 * k + 1  # 21
    
    # The ratio is scale-invariant, so we can fix ||c||_2 = 1 and minimize ||f||_4/||f||_2
    # Equivalently, minimize ||f||_4^4 / ||f||_2^2 (or any monotone function)
    
    # Strategy: The flattest function (most uniform |f|) minimizes the L4/L2 ratio.
    # For spherical harmonics of fixed degree, we want to spread the energy evenly.
    # 
    # Key insight: A single Y_0 (zonal harmonic) has a peaked distribution.
    # A combination that cancels out angular dependence would be ideal.
    #
    # For degree-10 harmonics, the "flattest" should be one that has minimal
    # variation over the sphere. Let's try optimization.
    
    # Objective: minimize R = ||f||_4 / ||f||_2
    # Since R is scale invariant, we can fix sum c_m^2 = 1 and minimize ||f||_4
    # (since ||f||_2 is then fixed by the orthonormality... wait, no, the basis is orthonormal
    #  so ||f||_2^2 = sum c_m^2 if the quadrature is exact for degree 20, which it is)
    
    # Actually with orthonormal basis: ||f||_2^2 = sum_m c_m^2 (exactly, since GL70 is exact for deg 40)
    # So if we normalize c to unit norm, ||f||_2 = 1, and we just minimize ||f||_4.
    # R = ||f||_4 / ||f||_2 = ||f||_4 when ||c||_2 = 1.
    
    # Let's do a multi-start local optimization.
    
    best_coeffs = None
    best_ratio = np.inf
    
    def objective(c):
        """Minimize ||f||_4 given ||c||_2 = 1 (handled by normalization outside)."""
        f = np.tensordot(c, B, axes=(0, 0))
        f4 = f**4
        return np.sum(W * f4)
    
    def objective_and_grad(c):
        """Return (objective, gradient) for ||f||_4^4 / 4 to make grad cleaner."""
        f = np.tensordot(c, B, axes=(0, 0))
        f4 = f**4
        obj = np.sum(W * f4)
        # d/dc_m sum W f^4 = sum W * 4 f^3 * B_m
        grad = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
        return obj, grad
    
    # Multi-start with various initializations
    np.random.seed(42)
    
    initializations = []
    
    # 1. All ones (baseline)
    initializations.append(np.ones(dim))
    
    # 2. Only m=0
    c0 = np.zeros(dim)
    c0[10] = 1.0
    initializations.append(c0.copy())
    
    # 3. Random unit vectors - more of them
    for trial in range(40):
        c = np.random.randn(dim)
        c /= np.linalg.norm(c)
        initializations.append(c)
    
    # 4. Alternating signs to promote cancellation
    c_alt = np.ones(dim)
    c_alt[::2] = -1.0
    initializations.append(c_alt / np.linalg.norm(c_alt))
    
    # 5. Try to approximate a constant: mix modes to reduce variance
    # The m=0 mode is the most "peaked", so downweight it
    c_low_m0 = np.ones(dim)
    c_low_m0[10] = 0.1  # reduce m=0
    initializations.append(c_low_m0 / np.linalg.norm(c_low_m0))
    
    # 6. Pure high-m modes (more oscillatory, might be flatter)
    for m_val in [5, 8, 10]:
        for sign in [1, -1]:
            c = np.zeros(dim)
            idx_pos = m_val + 10
            idx_neg = -m_val + 10
            c[idx_pos] = sign
            c[idx_neg] = sign
            initializations.append(c / np.linalg.norm(c))
    
    # 7. Structured initializations: try to make f more uniform
    # Equal weights on all modes
    c_equal = np.ones(dim)
    initializations.append(c_equal / np.linalg.norm(c_equal))
    
    # 8. Weighted by 1/sqrt(m+1) to reduce high-m contribution
    c_weighted = np.array([1.0/np.sqrt(abs(m_idx-10)+1) for m_idx in range(dim)])
    initializations.append(c_weighted / np.linalg.norm(c_weighted))
    
    # 9. Cosine-weighted: c_m = cos(m*pi/20)
    c_cos = np.array([np.cos((m_idx-10)*np.pi/20) for m_idx in range(dim)])
    initializations.append(c_cos / np.linalg.norm(c_cos))
    
    # 10. Sine-weighted: c_m = sin(m*pi/20) for m!=0
    c_sin = np.array([np.sin((m_idx-10)*np.pi/20) if m_idx != 10 else 0 for m_idx in range(dim)])
    if np.linalg.norm(c_sin) > 1e-10:
        initializations.append(c_sin / np.linalg.norm(c_sin))
    
    # 11. Try symmetric patterns with specific phase relationships
    for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        c_phase = np.array([np.cos((m_idx-10)*phase) for m_idx in range(dim)])
        initializations.append(c_phase / np.linalg.norm(c_phase))
        
        c_phase2 = np.array([np.sin((m_idx-10)*phase) for m_idx in range(dim)])
        if np.linalg.norm(c_phase2) > 1e-10:
            initializations.append(c_phase2 / np.linalg.norm(c_phase2))
    
    # 12. Try to minimize variance: use eigenvalue approach
    # Build the matrix M where M_ab = <Y_a * Y_b * Y_c * Y_d> type integrals
    # Actually, let's just try more random starts with different distributions
    for trial in range(20):
        # Use different random distributions
        c = np.random.exponential(1, size=dim)
        c /= np.linalg.norm(c)
        initializations.append(c)
        
        c = np.random.laplace(0, 1, size=dim)
        c /= np.linalg.norm(c)
        initializations.append(c)
    
    # Local optimization using L-BFGS-B with penalty for norm constraint
    def local_opt(c_init, max_iters=1000, lr=0.01):
        """Optimize using scipy L-BFGS-B with norm penalty."""
        c = c_init.copy()
        c = c / np.linalg.norm(c)
        
        def obj_penalty(c_vec):
            """Objective with penalty for deviating from unit norm."""
            norm_sq = np.dot(c_vec, c_vec)
            penalty = 1000.0 * (norm_sq - 1.0)**2
            f = np.tensordot(c_vec, B, axes=(0, 0))
            f4 = f**4
            base_obj = np.sum(W * f4)
            return base_obj + penalty
        
        def grad_penalty(c_vec):
            """Gradient of penalized objective."""
            norm_sq = np.dot(c_vec, c_vec)
            penalty_grad = 4000.0 * (norm_sq - 1.0) * c_vec
            f = np.tensordot(c_vec, B, axes=(0, 0))
            base_grad = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
            return base_grad + penalty_grad
        
        # Use L-BFGS-B
        result = minimize(
            obj_penalty,
            c,
            jac=grad_penalty,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        c_opt = result.x
        c_opt = c_opt / np.linalg.norm(c_opt)
        
        # Compute the actual ratio
        f = np.tensordot(c_opt, B, axes=(0, 0))
        f2 = f * f
        norm2_sq = np.sum(W * f2)
        f4 = f2 * f2
        norm4_4 = np.sum(W * f4)
        
        if norm2_sq > 0:
            ratio = np.sqrt(np.sqrt(norm4_4)) / np.sqrt(norm2_sq)
        else:
            ratio = np.inf
        
        return c_opt, ratio
    
    # Run local optimization from each initialization
    for i, c_init in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        c_opt, ratio = local_opt(c_init, max_iters=500)
        
        if ratio < best_ratio:
            best_ratio = ratio
            best_coeffs = c_opt.copy()
    
    # Additional refinement: try more random starts if time permits
    if time.time() < deadline - 30:
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            c = np.random.randn(dim)
            c /= np.linalg.norm(c)
            c_opt, ratio = local_opt(c, max_iters=300, lr=0.01)
            if ratio < best_ratio:
                best_ratio = ratio
                best_coeffs = c_opt.copy()
    
    # Final refinement with more iterations
    if best_coeffs is not None and time.time() < deadline - 10:
        c_final, ratio_final = local_opt(best_coeffs, max_iters=2000, lr=0.001)
        if ratio_final < best_ratio:
            best_ratio = ratio_final
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
