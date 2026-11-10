# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time


def _build_basis(k=10):
    """
    Build the 70x130 grid and precompute the 21 real orthonormal spherical
    harmonics Y_m(theta, phi) for m = -k..k on that grid, along with the
    quadrature weights.
    """
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl in (-1, 1), w_gl are weights for integral over [-1,1]
    # Surface measure: dOmega = dphi d(mu) where mu = cos(theta), integral over mu in [-1,1], phi in [0, 2pi)
    # So weight for each (mu_i, phi_j) point is w_gl[i] * (2*pi/130)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    # weights shape (70, 130)
    weights = np.outer(w_gl, np.ones(n_phi)) * dphi

    # Build harmonics
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # P_10^m with Condon-Shortley phase (scipy lpmv includes it)

    coeffs = np.zeros((2 * k + 1, 70, n_phi), dtype=np.float64)
    for m in range(-k, k + 1):
        am = abs(m)
        # Compute N_m
        # (10-am)!/(10+am)!
        # Use log to avoid overflow
        log_num = np.sum(np.log(np.arange(1, 11 - am))) if am < 10 else 0.0
        log_den = np.sum(np.log(np.arange(1, 11 + am))) if am > 0 else 0.0
        ratio = np.exp(log_num - log_den)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * ratio)

        P = lpmv(am, 10, x_gl)  # shape (70,)
        if m == 0:
            coeffs[m + k, :, :] = N_m * P[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            coeffs[m + k, :, :] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            coeffs[m + k, :, :] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.sin(am * phi)[np.newaxis, :]

    return coeffs, weights


def _compute_r(c, basis, weights):
    """
    Compute R = ||f||_4 / ||f||_2 for coefficient vector c.
    """
    # f = sum_m c_m * Y_m, shape (70, 130)
    f = np.tensordot(c, basis, axes=(0, 0))  # (70, 130)
    w = weights
    # ||f||_2^2 = sum f^2 * w
    norm2_sq = np.sum(f * f * w)
    # ||f||_4^4 = sum f^4 * w
    norm4_4 = np.sum(f * f * f * f * w)
    if norm2_sq <= 0:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2


def _objective(c, basis, weights):
    return _compute_r(c, basis, weights)


def _grad_r(c, basis, weights):
    """
    Compute gradient of R = ||f||_4 / ||f||_2 with respect to c.
    """
    f = np.tensordot(c, basis, axes=(0, 0))
    w = weights
    norm2_sq = np.sum(f * f * w)
    norm4_4 = np.sum(f * f * f * f * w)
    if norm2_sq <= 0:
        return np.zeros_like(c)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    R = norm4 / norm2

    # dR/dc_m = (1/norm2) * d||f||_4/dc_m - (norm4/norm2^2) * d||f||_2/dc_m
    # d||f||_4/dc_m = (1/4) * ||f||_4^{-3} * d(||f||_4^4)/dc_m = (1/4) * norm4^{-3} * sum(4 f^3 Y_m w) = norm4^{-3} * sum(f^3 Y_m w)
    # d||f||_2/dc_m = (1/2) * ||f||_2^{-1} * d(||f||_2^2)/dc_m = (1/2) * norm2^{-1} * sum(2 f Y_m w) = norm2^{-1} * sum(f Y_m w)

    # grad_norm4 = norm4^{-3} * (Y_m dot f^3 w)
    # grad_norm2 = norm2^{-1} * (Y_m dot f w)

    f3 = f * f * f
    dot_f3 = np.tensordot(basis, f3 * w, axes=([1, 2], [0, 1]))  # shape (21,)
    dot_f = np.tensordot(basis, f * w, axes=([1, 2], [0, 1]))  # shape (21,)

    grad_norm4 = dot_f3 / (norm4 ** 3)
    grad_norm2 = dot_f / norm2

    grad_R = grad_norm4 / norm2 - (norm4 / (norm2 ** 2)) * grad_norm2
    return grad_R


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 minimizing the L4/L2 ratio
    of the degree-10 spherical harmonic on the unit sphere.
    """
    t_start = time.time()
    deadline = 620.0  # safety margin before 630s kill

    k = 10
    basis, weights = _build_basis(k)

    # Initial guess: all ones
    c_best = np.ones(2 * k + 1, dtype=np.float64)
    r_best = _objective(c_best, basis, weights)

    # Try multiple random initializations and local optimization
    rng = np.random.default_rng(42)
    n_restarts = 30
    max_iter = 2000
    tol = 1e-12

    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break

        # Random initialization
        if restart == 0:
            c = c_best.copy()
        else:
            c = rng.normal(0, 1, size=2 * k + 1)

        # Normalize to avoid issues (the ratio is scale-invariant)
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            continue
        c = c / norm_c

        # Gradient descent with adaptive step size
        step_size = 0.1
        prev_r = _objective(c, basis, weights)

        for it in range(max_iter):
            if time.time() > deadline - 30.0:
                break

            grad = _grad_r(c, basis, weights)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < 1e-14:
                break

            # Line search
            alpha = step_size
            improved = False
            for _ in range(30):
                c_trial = c - alpha * grad
                r_trial = _objective(c_trial, basis, weights)
                if r_trial < prev_r:
                    c = c_trial
                    prev_r = r_trial
                    improved = True
                    break
                alpha *= 0.5

            if not improved:
                # Try smaller step
                alpha = 1e-6
                c_trial = c - alpha * grad
                r_trial = _objective(c_trial, basis, weights)
                if r_trial < prev_r:
                    c = c_trial
                    prev_r = r_trial
                else:
                    break

            # Update step size
            if improved:
                step_size = min(step_size * 1.2, 1.0)
            else:
                step_size = max(step_size * 0.5, 1e-8)

        # Track best
        r_final = _objective(c, basis, weights)
        if r_final < r_best:
            r_best = r_final
            c_best = c.copy()

    # Ensure valid output
    if not np.all(np.isfinite(c_best)):
        c_best = np.ones(2 * k + 1, dtype=np.float64)
    max_abs = np.max(np.abs(c_best))
    if max_abs < 1e-3:
        c_best = c_best / max_abs * 1.0
    elif max_abs > 1e5:
        c_best = c_best / max_abs * 1.0

    return c_best


# EVOLVE-BLOCK-END
