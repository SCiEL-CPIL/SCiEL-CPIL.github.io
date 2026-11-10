# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize
from scipy.integrate import gauss_legendre

# Precompute the spherical harmonic basis and quadrature weights
# at module level so they are computed once per import/evaluation.

_K = 10
_N_THETA = 70
_N_PHI = 130

def _build_basis_and_quadrature():
    """
    Build:
      - theta_quadrature points (x_i = cos theta_i) and weights w_i for
        70-point Gauss-Legendre on [-1, 1].
      - phi_j = 2*pi*j/N_PHI for j=0..N_PHI-1.
      - basis[i, j, m] = Y_m(x_i, phi_j) for m = -10..10,
        where Y_0 = N_0 P_10(cos theta),
              Y_m = sqrt(2) N_|m| P_10^{|m|}(cos theta) cos(m phi) for m>0,
              Y_m = sqrt(2) N_|m| P_10^{|m|}(cos theta) sin(|m| phi) for m<0.
      - total_weight[i, j] = w_i * (2*pi / N_PHI)
        (the quadrature weight for the surface integral).
    """
    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = gauss_legendre(_N_THETA)
    # x = cos(theta_i), w_i are the GL weights.
    # The surface integral is ∫_{-1}^{1} ∫_0^{2π} f(cosθ, φ) dφ dx
    # = sum_i w_i sum_j f(x_i, φ_j) * (2π / N_PHI)
    dphi = 2.0 * np.pi / _N_PHI
    phi = np.arange(_N_PHI) * dphi
    phi = phi % (2.0 * np.pi)  # ensure in [0, 2π)

    # Normalization constants N_m = sqrt(21/(4π) * (10-m)!/(10+m)!)
    # Use log-gamma for numerical stability.
    from scipy.special import gammaln
    N = np.zeros(2 * _K + 1)
    for m_abs in range(_K + 1):
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + gammaln(_K - m_abs + 1) - gammaln(_K + m_abs + 1))
        N[m_abs] = np.exp(log_N)
    # N indexed by |m|; we'll map m -> |m|

    # Build basis: shape (N_THETA, N_PHI, 2*K+1)
    basis = np.zeros((_N_THETA, _N_PHI, 2 * _K + 1), dtype=np.float64)

    # For each m, compute P_10^{|m|}(x) for all x, then fill in cos/sin parts.
    # m ranges from -10 to 10. Index in array: m_idx = m + 10.
    for m in range(-_K, _K + 1):
        m_idx = m + _K
        m_abs = abs(m)
        # Associated Legendre P_10^{m_abs}(x) with Condon-Shortley phase.
        # scipy.special.lpmv(m, n, x) gives P_n^m(x) with Condon-Shortley.
        P = lpmv(m_abs, _K, x)  # shape (N_THETA,)
        if m == 0:
            Y_theta = N[0] * P  # N_0 * P_10^0
            basis[:, :, m_idx] = np.outer(Y_theta, np.ones(_N_PHI))
        elif m > 0:
            Y_theta = np.sqrt(2.0) * N[m_abs] * P
            basis[:, :, m_idx] = np.outer(Y_theta, np.cos(m * phi))
        else:
            Y_theta = np.sqrt(2.0) * N[m_abs] * P
            basis[:, :, m_idx] = np.outer(Y_theta, np.sin(m_abs * phi))

    # Total weight for each (i, j)
    total_weight = np.outer(w, np.full(_N_PHI, dphi))  # shape (N_THETA, N_PHI)

    return x, w, phi, basis, total_weight

# Precompute
_X, _W, _PHI, _BASIS, _TOTAL_W = _build_basis_and_quadrature()


def _compute_R(coeffs):
    """
    Compute R = ||f||_4 / ||f||_2 using the precomputed quadrature.
    coeffs: shape (21,), ordered m = -10..10.
    Returns scalar R.
    """
    # f(i,j) = sum_m c_m * basis[i, j, m]
    # basis shape: (N_THETA, N_PHI, 21), coeffs shape: (21,)
    f = np.tensordot(_BASIS, coeffs, axes=([2], [0]))  # shape (N_THETA, N_PHI)
    # f^2 and f^4
    f2 = f * f
    f4 = f2 * f2
    # Integrals: ∫ f^2 dΩ ≈ sum_{i,j} total_weight[i,j] * f[i,j]^2
    int_f2 = np.sum(_TOTAL_W * f2)
    int_f4 = np.sum(_TOTAL_W * f4)
    # Norms: ||f||_2 = sqrt(int_f2), ||f||_4 = (int_f4)^(1/4)
    norm2 = np.sqrt(int_f2)
    norm4 = int_f4 ** 0.25
    if norm2 < 1e-15:
        return np.inf
    return norm4 / norm2


def _compute_R_and_grad(coeffs):
    """
    Compute R and its gradient w.r.t. coeffs.
    """
    f = np.tensordot(_BASIS, coeffs, axes=([2], [0]))
    f2 = f * f
    f4 = f2 * f2
    int_f2 = np.sum(_TOTAL_W * f2)
    int_f4 = np.sum(_TOTAL_W * f4)
    norm2 = np.sqrt(int_f2)
    norm4 = int_f4 ** 0.25
    if norm2 < 1e-15:
        return np.inf, np.zeros_like(coeffs)
    R = norm4 / norm2

    # Gradient of R = ||f||_4 / ||f||_2
    # dR/dc_m = (1/norm2) * d||f||_4/dc_m - (norm4/norm2^2) * d||f||_2/dc_m
    # d||f||_4/dc_m = (1/4) * ||f||_4^{-3} * d(int_f4)/dc_m
    # d(int_f4)/dc_m = sum_{i,j} w_ij * 4 f[i,j]^3 * basis[i,j,m]
    # d||f||_2/dc_m = (1/norm2) * d(int_f2)/dc_m
    # d(int_f2)/dc_m = sum_{i,j} w_ij * 2 f[i,j] * basis[i,j,m]

    # d_int_f4/dc_m = 4 * sum_{i,j} w_ij * f^3 * basis_m
    # d_int_f2/dc_m = 2 * sum_{i,j} w_ij * f * basis_m

    # We can compute these as: 
    # d_int_f4 = 4 * (f^3 * W) @ basis  (where W is total_weight broadcast)
    # d_int_f2 = 2 * (f * W) @ basis

    weighted_f3 = (_TOTAL_W * f * f2)  # W * f^3
    weighted_f = (_TOTAL_W * f)        # W * f

    # d_int_f4 shape: (21,) = sum over i,j of weighted_f3[i,j] * basis[i,j,m]
    d_int_f4 = np.tensordot(weighted_f3, _BASIS, axes=([0, 1], [0, 1]))
    d_int_f2 = np.tensordot(weighted_f, _BASIS, axes=([0, 1], [0, 1]))

    d_norm4 = (0.25) * (int_f4 ** (-0.75)) * d_int_f4  # (1/4) * int_f4^{-3/4} * d_int_f4
    d_norm2 = (0.5) * (int_f2 ** (-0.5)) * d_int_f2   # (1/(2 sqrt(int_f2))) * d_int_f2

    grad_R = d_norm4 / norm2 - (norm4 / (norm2 * norm2)) * d_norm2

    return R, grad_R


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m (m=-10..10) that minimize R = ||f||_4 / ||f||_2
    for f = sum c_m Y_m, degree-10 real orthonormal spherical harmonics.
    Returns np.ndarray of shape (21,).
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 s wall-clock limit; keep safety margin
    safety_margin = 30.0  # seconds

    best_R = np.inf
    best_coeffs = np.ones(21, dtype=np.float64)

    # Strategy: multi-start L-BFGS with analytic gradient.
    # Generate diverse initial points, run L-BFGS on each, keep best.

    rng = np.random.default_rng(42)

    # Precompute basis for fast evaluation (already done at module level)

    def objective_and_grad(c):
        # Scale-invariant: R is homogeneous of degree 0, so we can fix ||c||_2 = 1
        # to avoid the flat direction. But L-BFGS on the ratio directly works too.
        # Let's use the ratio directly; L-BFGS should handle it.
        val, grad = _compute_R_and_grad(c)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad

    # Generate initial points
    n_restarts = 50  # We'll adjust based on time
    inits = []

    # All ones
    inits.append(np.ones(21, dtype=np.float64))

    # Random directions
    for _ in range(n_restarts - 1):
        v = rng.standard_normal(21)
        norm = np.linalg.norm(v)
        if norm > 1e-15:
            v /= norm
        inits.append(v)

    # Also try some structured inits: single m, pairs of m, etc.
    for m in range(21):
        v = np.zeros(21, dtype=np.float64)
        v[m] = 1.0
        inits.append(v)

    # Pairwise: m and -m
    for m in range(1, 11):
        v = np.zeros(21, dtype=np.float64)
        v[m - 1 + 10] = 1.0  # m positive
        v[-m + 10] = 1.0     # m negative
        norm = np.linalg.norm(v)
        if norm > 1e-15:
            v /= norm
        inits.append(v)

    # Shuffle inits
    idx = np.arange(len(inits))
    rng.shuffle(idx)
    inits = [inits[i] for i in idx]

    # Run L-BFGS on each init, with time check
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline - safety_margin:
            break

        try:
            result = minimize(
                objective_and_grad,
                init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            R_val = result.fun
            if np.isfinite(R_val) and R_val < best_R:
                best_R = R_val
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Final polish: run more iterations on the best found
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin and best_R < np.inf:
        try:
            result = minimize(
                objective_and_grad,
                best_coeffs,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            R_val = result.fun
            if np.isfinite(R_val) and R_val < best_R:
                best_R = R_val
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Validate
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = np.ones(21, dtype=np.float64)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs

    return best_coeffs


# EVOLVE-BLOCK-END
