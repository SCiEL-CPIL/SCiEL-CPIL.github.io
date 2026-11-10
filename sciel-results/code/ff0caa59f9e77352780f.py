# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre, gammaln
from scipy.optimize import minimize


def _precompute():
    """Precompute quadrature nodes, weights, and spherical harmonic basis."""
    k = 10

    # Gauss-Legendre quadrature for cos(theta) in [-1, 1]
    n_theta = 70
    x, w = np.polynomial.legendre.leggauss(n_theta)

    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    # Precompute P_10^m(cos(theta)) for m = 0..10 at all quadrature nodes
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = eval_legendre(10, m, x)

    # Normalization constants
    N = np.zeros(11)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + gammaln(11 - m) - gammaln(11 + m))
        N[m] = np.exp(log_N)

    # Build basis matrix: B[m_idx, i, j] where m_idx=0..20 corresponds to m=-10..10
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


def _evaluate_ratio(coeffs, B, W):
    """Compute ||f||_4 / ||f||_2 using precomputed basis and weights."""
    f = np.tensordot(coeffs, B, axes=(0, 0))
    f2 = f * f
    norm2_sq = np.sum(W * f2)
    f4 = f2 * f2
    norm4_4 = np.sum(W * f4)

    if norm2_sq <= 0:
        return np.inf

    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))

    return norm4 / norm2


def _objective_and_grad(c, B, W):
    """Return objective (||f||_4^4) and its gradient for given coefficients."""
    f = np.tensordot(c, B, axes=(0, 0))
    f4 = f**4
    obj = np.sum(W * f4)
    grad = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
    return obj, grad


def _objective_sphere(c, B, W):
    """Objective on the unit sphere: minimize ||f||_4^4 given ||c||=1."""
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-15:
        return 1e10
    c_unit = c / c_norm
    f = np.tensordot(c_unit, B, axes=(0, 0))
    f4 = f**4
    return np.sum(W * f4)


def _objective_sphere_grad(c, B, W):
    """Objective and gradient on the unit sphere."""
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-15:
        return 1e10, np.zeros_like(c)
    c_unit = c / c_norm
    f = np.tensordot(c_unit, B, axes=(0, 0))
    f4 = f**4
    obj = np.sum(W * f4)
    # Gradient with respect to c (before normalization)
    # d/dc sum W f^4 where f = B^T c_unit
    # = 4 sum W f^3 B_m / ||c|| - (c/||c||^2) * (sum W f^4) ... 
    # Actually let's just compute gradient of obj(c_unit) w.r.t c
    # Using chain rule: d/dc = (I - c c^T / ||c||^2) @ d/dc_unit
    grad_unit = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
    # Project to tangent space
    grad = grad_unit - np.dot(grad_unit, c_unit) * c_unit
    return obj, grad


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimize the norm ratio ||f||_4 / ||f||_2 on the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin

    B, W, x, phi, n_theta, n_phi = _precompute()

    dim = 21

    def check_time():
        return time.time() < deadline - 10

    best_coeffs = None
    best_ratio = np.inf

    # Strategy: Use L-BFGS-B on the sphere with multiple restarts
    # Since the ratio is scale-invariant, we optimize on the unit sphere
    
    def make_objective(c):
        """Return objective value for L-BFGS-B (uses normalized c)."""
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            return 1e10
        c_unit = c / c_norm
        f = np.tensordot(c_unit, B, axes=(0, 0))
        f4 = f**4
        return np.sum(W * f4)

    def make_grad(c):
        """Return gradient for L-BFGS-B."""
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            return np.zeros(dim)
        c_unit = c / c_norm
        f = np.tensordot(c_unit, B, axes=(0, 0))
        grad_unit = 4.0 * np.tensordot(W * f**3, B, axes=([0, 1], [1, 2]))
        # Project to tangent space of sphere
        grad = grad_unit - np.dot(grad_unit, c_unit) * c_unit
        return grad

    # Generate diverse initializations
    np.random.seed(12345)
    initializations = []

    # 1. All ones
    initializations.append(np.ones(dim) / np.sqrt(dim))

    # 2. Only m=0
    c0 = np.zeros(dim)
    c0[10] = 1.0
    initializations.append(c0.copy())

    # 3. Random unit vectors
    for trial in range(30):
        c = np.random.randn(dim)
        c /= np.linalg.norm(c)
        initializations.append(c)

    # 4. Alternating signs
    c_alt = np.ones(dim)
    c_alt[::2] = -1.0
    initializations.append(c_alt / np.linalg.norm(c_alt))

    # 5. Reduced m=0
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

    # 7. Mix of modes with random weights
    for trial in range(20):
        c = np.random.randn(dim)
        c[10] *= 0.3  # downweight m=0
        c /= np.linalg.norm(c)
        initializations.append(c)

    # Run L-BFGS-B from each initialization
    for i, c_init in enumerate(initializations):
        if not check_time():
            break

        try:
            result = minimize(
                make_objective,
                c_init,
                jac=make_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )

            c_opt = result.x
            c_norm = np.linalg.norm(c_opt)
            if c_norm < 1e-15:
                continue
            c_opt = c_opt / c_norm

            ratio = _evaluate_ratio(c_opt, B, W)

            if ratio < best_ratio:
                best_ratio = ratio
                best_coeffs = c_opt.copy()

        except Exception:
            continue

    # Additional refinement with more iterations on the best solution
    if best_coeffs is not None and check_time():
        try:
            result = minimize(
                make_objective,
                best_coeffs,
                jac=make_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            c_opt = result.x
            c_norm = np.linalg.norm(c_opt)
            if c_norm >= 1e-15:
                c_opt = c_opt / c_norm
                ratio = _evaluate_ratio(c_opt, B, W)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Try even more random restarts if time permits
    if check_time():
        for trial in range(50):
            if not check_time():
                break
            c = np.random.randn(dim)
            c /= np.linalg.norm(c)
            try:
                result = minimize(
                    make_objective,
                    c,
                    jac=make_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                c_opt = result.x
                c_norm = np.linalg.norm(c_opt)
                if c_norm >= 1e-15:
                    c_opt = c_opt / c_norm
                    ratio = _evaluate_ratio(c_opt, B, W)
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_coeffs = c_opt.copy()
            except Exception:
                continue

    if best_coeffs is None:
        best_coeffs = np.ones(dim) / np.sqrt(dim)

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
