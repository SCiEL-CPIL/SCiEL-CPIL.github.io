# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_K = 10
_NTH = 70
_NPHI = 130
_DEADLINE = 590.0


def _precompute():
    """Precompute quadrature weights and the harmonic matrix H (21 x N)."""
    # Gauss-Legendre nodes/weights in u = cos(theta) in [-1, 1]
    u, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    # Surface measure weight: dOmega = dphi du.  Integrate over phi analytically
    # (orthogonality), leaving a 1D integral over u with weight w.
    # I_m = (1/2) * int_{-1}^{1} (Y_m(u,phi))^2 du  for m>0, (Y_0)^2 for m=0.
    # For m>0: Y_m = sqrt(2)*N_m*P_m*cos(m phi) -> phi-integral of cos^2 = pi
    #          -> I_m = pi * N_m^2 * int P_m^2 du
    # For m=0: Y_0 = N_0 P_0 -> I_0 = (1/2) N_0^2 int P_0^2 du

    # Build H: H[m, (i,j)] = Y_m(theta_i, phi_j)
    Npts = _NTH * _NPHI
    H = np.empty((_NPHI, _NTH, 21), dtype=np.float64)  # [phi, theta, m]
    for m in range(21):
        mm = m - _K
        if mm == 0:
            N0 = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * factorial(_K) / factorial(_K))
            P0 = lpmv(0, _K, u)
            H[:, :, 0] = N0 * P0[None, :]
        elif mm > 0:
            Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * factorial(_K - mm) / factorial(_K + mm))
            Pm = lpmv(mm, _K, u)
            H[:, :, mm] = np.sqrt(2.0) * Nm * Pm[None, :] * np.cos(mm * phi)[:, None]
        else:
            a = -mm
            Na = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * factorial(_K - a) / factorial(_K + a))
            Pa = lpmv(a, _K, u)
            H[:, :, mm] = np.sqrt(2.0) * Na * Pa[None, :] * np.sin(a * phi)[:, None]

    # Flatten to [Npts, 21], Npts = _NPHI * _NTH
    Hf = H.transpose(1, 2, 0).reshape(_NTH * _NPHI, 21)  # [theta, m, phi] -> [theta*phi, m]

    # Weights for the 1D-over-u integration (phi integrated analytically).
    # f(u,phi) = sum_m c_m Y_m.  f^2 and f^4 integrals over phi use orthogonality.
    # We'll keep the full 2D grid for generality (70*130=9100 pts, cheap).
    # Quadrature weight per (theta,phi) point:
    #   dOmega = dphi * du,  so weight = w_i * dphi
    W = np.tile(w * dphi, _NPHI)  # [Npts]

    return Hf, W, u, w, phi, dphi


def _compute_R(c):
    """Compute R = ||f||_4 / ||f||_2 via quadrature.  Returns (R, f_vals)."""
    f = c @ Hf  # [Npts]
    f2 = f * f
    f4 = f2 * f2
    I2 = float(np.dot(W, f2))
    I4 = float(np.dot(W, f4))
    if I2 <= 0.0:
        return np.inf, f
    R2 = I4 / I2  # = (||f||_4^4)/(||f||_2^4)
    R = np.sqrt(R2)  # = ||f||_4 / ||f||_2
    return R, f


def _objective_and_grad(c):
    """Objective = R^4 (monotone in R) and its analytic gradient."""
    f = c @ Hf
    f2 = f * f
    f4 = f2 * f2
    I2 = float(np.dot(W, f2))
    I4 = float(np.dot(W, f4))
    if I2 <= 0.0:
        return np.inf, np.zeros(21)
    # R^4 = I4^2 / I2^2
    obj = I4 * I4 / (I2 * I2)
    # d obj / d c_m = d/dc_m [ I4^2 / I2^2 ]
    #   = (2 I4 I4' I2^2 - I4^2 * 2 I2 I2') / I2^4
    #   = 2 I4 I4' / I2^2 - 2 I4^2 I2' / I2^3
    # I2' = d/dc_m sum W f2 = 2 sum W f Y_m = 2 (f * Y_m weighted)
    # I4' = d/dc_m sum W f4 = 4 sum W f3 Y_m
    # Let A = sum W f Y_m  (vector over m), B = sum W f^3 Y_m
    # I2' = 2 A, I4' = 4 B
    # grad = 2 I4 * 4B / I2^2 - 2 I4^2 * 2A / I2^3
    #      = 8 I4 B / I2^2 - 4 I4^2 A / I2^3
    A = Hf.T @ (W * f)          # [21]
    B = Hf.T @ (W * f3) if False else Hf.T @ (W * (f * f2))  # [21]
    grad = 8.0 * I4 * B / (I2 * I2) - 4.0 * I4 * I4 * A / (I2 * I2 * I2)
    return obj, grad


def _make_seeds():
    """Generate a list of diverse starting points."""
    seeds = []
    # 1. All ones
    seeds.append(np.ones(21))
    # 2. Symmetric: c_m = c_{-m} (cosine-like), only m>=0 nonzero, m=0 separate
    s = np.zeros(21)
    s[10] = 1.0
    for m in range(1, 11):
        s[10 + m] = 1.0
        s[10 - m] = 1.0
    seeds.append(s)
    # 3. Antisymmetric: c_m = -c_{-m}, sine-like
    s = np.zeros(21)
    for m in range(1, 11):
        s[10 + m] = 1.0
        s[10 - m] = -1.0
    seeds.append(s)
    # 4. Single m = 0
    s = np.zeros(21); s[10] = 1.0; seeds.append(s)
    # 5. Single m = 10
    s = np.zeros(21); s[20] = 1.0; seeds.append(s)
    # 6. Low-order emphasis: weight decreasing with |m|
    s = np.zeros(21)
    for m in range(21):
        mm = abs(m - 10)
        s[m] = 1.0 / (1.0 + mm)
    seeds.append(s)
    # 7. High-order emphasis
    s = np.zeros(21)
    for m in range(21):
        mm = abs(m - 10)
        s[m] = 1.0 + mm
    seeds.append(s)
    # 8. Alternating sign
    s = np.zeros(21)
    for m in range(21):
        mm = m - 10
        s[m] = 1.0 if mm % 2 == 0 else -1.0
    seeds.append(s)
    # 9. Random seeds (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    for _ in range(16):
        s = rng.standard_normal(21)
        seeds.append(s)
    return seeds


def _try_optimize(c0):
    """Run L-BFGS-B from c0, return best c found."""
    try:
        obj0, _ = _objective_and_grad(c0)
        res = minimize(
            _objective_and_grad,
            c0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10, 'maxls': 50},
        )
        c = res.x
        obj, _ = _objective_and_grad(c)
        if np.isfinite(obj) and obj < obj0:
            return c
        return c0
    except Exception:
        return c0


def rudin_problem10() -> np.ndarray:
    global Hf, W
    t0 = time.time()
    Hf, W, u, w, phi, dphi = _precompute()

    seeds = _make_seeds()
    best_c = None
    best_R = np.inf

    for i, c0 in enumerate(seeds):
        if time.time() - t0 > _DEADLINE - 30.0:
            break
        c_opt = _try_optimize(c0)
        R, _ = _compute_R(c_opt)
        if np.isfinite(R) and R < best_R:
            best_R = R
            best_c = c_opt.copy()

    # Random restarts with remaining time
    rng = np.random.default_rng(12345)
    n_extra = 0
    while time.time() - t0 < _DEADLINE - 30.0:
        c0 = rng.standard_normal(21)
        # Scale to unit norm for stability
        nrm = np.linalg.norm(c0)
        if nrm > 0:
            c0 = c0 / nrm
        c_opt = _try_optimize(c0)
        R, _ = _compute_R(c_opt)
        if np.isfinite(R) and R < best_R:
            best_R = R
            best_c = c_opt.copy()
        n_extra += 1
        # If we're making no progress and time is running out, stop
        if n_extra > 200:
            break

    if best_c is None:
        best_c = np.ones(21, dtype=np.float64)

    # Ensure validity: max|c| in [1e-3, 1e5]
    mx = np.max(np.abs(best_c))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    elif mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c.astype(np.float64)


# EVOLVE-BLOCK-END
