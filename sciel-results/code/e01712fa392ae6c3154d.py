# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre, gammaln
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def _precompute():
    """Precompute quadrature nodes, weights, and spherical harmonic basis."""
    k = 10

    n_theta = 70
    x, w = np.polynomial.legendre.leggauss(n_theta)

    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    # Build basis matrix B[m_idx, i, j]
    # m_idx = 0..20 corresponds to m = -10..10
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = eval_legendre(10, m, x)

    N = np.zeros(11)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + gammaln(11 - m) - gammaln(11 + m))
        N[m] = np.exp(log_N)

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
    W = np.outer(w, np.ones(n_phi)) * dphi  # shape (n_theta, n_phi)

    # Flatten B and W for matrix operations
    # B_flat: (21, n_theta * n_phi)
    B_flat = B.reshape(21, n_theta * n_phi)
    W_flat = W.reshape(n_theta * n_phi)

    # Precompute the 4th-order tensor contraction for ||f||_4^4
    # ||f||_4^4 = sum_{i,j} W_{ij} (sum_a c_a B_{a,ij})^4
    # = sum_{a,b,c,d} c_a c_b c_c c_d * T_{abcd}
    # where T_{abcd} = sum_{i,j} W_{ij} B_{a,ij} B_{b,ij} B_{c,ij} B_{d,ij}
    #
    # For efficient computation, we can use the identity:
    # sum_{i,j} W_{ij} B_{a,ij} B_{b,ij} B_{c,ij} B_{d,ij}
    # = sum_{i,j} W_{ij} (B_{a,ij} B_{b,ij}) (B_{c,ij} B_{d,ij})
    #
    # Let M_{ab, cd} = sum_{i,j} W_{ij} B_{a,ij} B_{b,ij} B_{c,ij} B_{d,ij}
    # This is a 4-index tensor of shape (21, 21, 21, 21)
    # We can compute it as: for each quadrature point (i,j), form outer products

    # More efficient: use the fact that ||f||_4^4 = sum_k W_k f_k^4
    # where f_k = sum_a c_a B_{a,k} (flattened)
    # = c^T (sum_k W_k B_k B_k^T B_k B_k^T) c ... no that's not right either
    # Actually: f_k^4 = (c^T B_k)^4 = sum_{abcd} c_a c_b c_c c_d B_{a,k} B_{b,k} C_{c,k} B_{d,k}
    # So ||f||_4^4 = sum_{abcd} c_a c_b c_c c_d * sum_k W_k B_{a,k} B_{b,k} B_{c,k} B_{d,k}

    # Precompute T as a 4D array or use a different approach
    # For 21x21x21x21 = ~1.9M entries, this is feasible

    # Compute T_{abcd} = sum_k W_k B_{a,k} B_{b,k} B_{c,k} B_{d,k}
    # Using einsum for efficiency
    # B_flat is (21, n_points)
    n_points = n_theta * n_phi
    # T = np.einsum('ak,bk,ck,dk,k->abcd', B_flat, B_flat, B_flat, B_flat, W_flat)
    # This might be slow. Let's use a different approach.

    # Alternative: compute the 2D matrix Q where Q_{ab} = sum_k W_k B_{a,k} B_{b,k} (this is just W-weighted inner product)
    # But we need the 4th power, not 2nd.

    # Let's just use the direct computation with vectorized operations
    # f = B_flat.T @ c  (n_points,)
    # ||f||_4^4 = sum W * f^4
    # grad = 4 * sum W * f^3 * B_flat (for each coefficient)

    # For the optimization, we'll use the direct approach but with precomputed
    # B_flat and W_flat for speed.

    return B, W, B_flat, W_flat, x, phi, n_theta, n_phi, n_points


def _compute_ratio(c, B_flat, W_flat):
    """Compute ||f||_4 / ||f||_2 for coefficient vector c."""
    f = B_flat.T @ c  # shape (n_points,)
    f2 = f * f
    norm2_sq = np.dot(W_flat, f2)
    if norm2_sq <= 0:
        return np.inf
    f4 = f2 * f2
    norm4_4 = np.dot(W_flat, f4)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2


def _objective_and_grad(c, B_flat, W_flat):
    """
    Objective: minimize ||f||_4^4 subject to ||c||_2^2 = 1.
    Using Lagrange multiplier approach: minimize ||f||_4^4 - lambda*(||c||_2^2 - 1)
    But since we normalize c, we just minimize ||f||_4^4 with c normalized.
    
    Returns (objective_value, gradient) for the unnormalized objective ||f||_4^4.
    """
    f = B_flat.T @ c  # (n_points,)
    f2 = f * f
    f4 = f2 * f2
    obj = np.dot(W_flat, f4)
    
    # Gradient: d/dc_m sum_k W_k f_k^4 = sum_k W_k * 4 * f_k^3 * B_{m,k}
    # = 4 * B_flat @ (W_flat * f^3)
    grad = 4.0 * B_flat @ (W_flat * f**3)
    
    return obj, grad


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimize the norm ratio ||f||_4 / ||f||_2 on the unit sphere.
    
    Returns:
        coeffs: np.ndarray of shape (21,) with real coefficients c_m,
            ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin

    B, W, B_flat, W_flat, x, phi, n_theta, n_phi, n_points = _precompute()

    dim = 21

    best_coeffs = None
    best_ratio = np.inf

    def _local_optimize(c_init, max_iter=2000, tol=1e-12):
        """
        Optimize using projected gradient / BFGS on the sphere.
        We minimize ||f||_4^4 subject to ||c||_2 = 1.
        Use L-BFGS-B with a penalty for the norm constraint, or use
        a retraction-based approach.
        
        Here we use scipy's minimize with a norm penalty.
        """
        c = c_init.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-15:
            c = np.random.randn(dim)
            c /= np.linalg.norm(c)
        else:
            c /= norm_c

        def penalty_obj(c_vec):
            norm_sq = np.dot(c_vec, c_vec)
            penalty = 10000.0 * (norm_sq - 1.0)**2
            f = B_flat.T @ c_vec
            f4 = f**4
            base = np.dot(W_flat, f4)
            return base + penalty

        def penalty_grad(c_vec):
            norm_sq = np.dot(c_vec, c_vec)
            p_grad = 40000.0 * (norm_sq - 1.0) * c_vec
            f = B_flat.T @ c_vec
            base_grad = 4.0 * B_flat @ (W_flat * f**3)
            return base_grad + p_grad

        result = minimize(
            penalty_obj,
            c,
            jac=penalty_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
        )

        c_opt = result.x
        c_opt = c_opt / np.linalg.norm(c_opt)

        ratio = _compute_ratio(c_opt, B_flat, W_flat)
        return c_opt, ratio

    # Generate diverse initializations
    np.random.seed(42)
    initializations = []

    # 1. All ones
    initializations.append(np.ones(dim) / np.sqrt(dim))

    # 2. Only m=0
    c0 = np.zeros(dim)
    c0[10] = 1.0
    initializations.append(c0.copy())

    # 3. Random unit vectors
    for _ in range(50):
        c = np.random.randn(dim)
        c /= np.linalg.norm(c)
        initializations.append(c)

    # 4. Alternating signs
    c_alt = np.ones(dim)
    c_alt[::2] = -1.0
    initializations.append(c_alt / np.linalg.norm(c_alt))

    # 5. Reduced m=0
    c_low = np.ones(dim)
    c_low[10] = 0.05
    initializations.append(c_low / np.linalg.norm(c_low))

    # 6. Pure high-m modes
    for m_val in [5, 8, 10]:
        for sign in [1, -1]:
            c = np.zeros(dim)
            c[m_val + 10] = sign
            c[-m_val + 10] = sign
            initializations.append(c / np.linalg.norm(c))

    # 7. Cosine-weighted
    c_cos = np.array([np.cos((m_idx - 10) * np.pi / 20) for m_idx in range(dim)])
    initializations.append(c_cos / np.linalg.norm(c_cos))

    # 8. Sine-weighted
    c_sin = np.array([np.sin((m_idx - 10) * np.pi / 20) if m_idx != 10 else 0.0 for m_idx in range(dim)])
    if np.linalg.norm(c_sin) > 1e-10:
        initializations.append(c_sin / np.linalg.norm(c_sin))

    # 9. Phase patterns
    for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        c_ph = np.array([np.cos((m_idx - 10) * phase) for m_idx in range(dim)])
        initializations.append(c_ph / np.linalg.norm(c_ph))
        c_ph2 = np.array([np.sin((m_idx - 10) * phase) for m_idx in range(dim)])
        if np.linalg.norm(c_ph2) > 1e-10:
            initializations.append(c_ph2 / np.linalg.norm(c_ph2))

    # 10. Exponential and Laplace distributions
    for _ in range(20):
        c = np.random.exponential(1, size=dim)
        initializations.append(c / np.linalg.norm(c))
        c = np.random.laplace(0, 1, size=dim)
        initializations.append(c / np.linalg.norm(c))

    # 11. Structured: try to make f as flat as possible
    # The zonal harmonic P_10(cos theta) has large peaks.
    # Mixing with high-m modes can help cancel angular variation.
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        c_struct = np.zeros(dim)
        c_struct[10] = alpha  # m=0 component
        # Distribute remaining weight among other modes
        remaining = 1.0 - alpha
        for m_idx in range(dim):
            if m_idx != 10:
                c_struct[m_idx] = remaining / (dim - 1) * np.random.choice([-1, 1])
        initializations.append(c_struct / np.linalg.norm(c_struct))

    # Run local optimization from each initialization
    for i, c_init in enumerate(initializations):
        if time.time() > deadline - 20:
            break

        c_opt, ratio = _local_optimize(c_init, max_iter=1500, tol=1e-12)

        if ratio < best_ratio:
            best_ratio = ratio
            best_coeffs = c_opt.copy()

    # Additional random restarts if time permits
    if time.time() < deadline - 40:
        for trial in range(40):
            if time.time() > deadline - 20:
                break
            c = np.random.randn(dim)
            c /= np.linalg.norm(c)
            c_opt, ratio = _local_optimize(c, max_iter=1000, tol=1e-12)
            if ratio < best_ratio:
                best_ratio = ratio
                best_coeffs = c_opt.copy()

    # Final high-precision polish
    if best_coeffs is not None and time.time() < deadline - 15:
        c_final, ratio_final = _local_optimize(best_coeffs, max_iter=5000, tol=1e-14)
        if ratio_final < best_ratio:
            best_ratio = ratio_final
            best_coeffs = c_final

    if best_coeffs is None:
        best_coeffs = np.ones(dim) / np.sqrt(dim)

    # Ensure validity
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (1e-3 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e5 / max_abs)

    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(dim) / np.sqrt(dim)

    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
