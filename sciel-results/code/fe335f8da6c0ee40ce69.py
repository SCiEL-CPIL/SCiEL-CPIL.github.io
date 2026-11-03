"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    start_time = time.time()
    deadline = 180  # seconds, keep margin for evaluator kill at 210s

    # Setup grid
    N = 4096
    x = np.linspace(-20, 20, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Precompute Hermite polynomials for the grid
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)

    # Effective basis vectors enforcing g(0) = 0
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    # Precompute Fourier transforms of basis vectors
    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx

    # Precompute quadratic form matrices
    # M for norm^2 = c^T M c
    M = np.zeros((3, 3))
    M[0, 0] = np.sum(v0**2) * dx
    M[1, 1] = np.sum(v4**2) * dx
    M[2, 2] = np.sum(v8**2) * dx
    M[0, 1] = M[1, 0] = np.sum(v0 * v4) * dx
    M[0, 2] = M[2, 0] = np.sum(v0 * v8) * dx
    M[1, 2] = M[2, 1] = np.sum(v4 * v8) * dx

    # Mx for x^2 norm = c^T Mx c
    Mx = np.zeros((3, 3))
    Mx[0, 0] = np.sum(x**2 * v0**2) * dx
    Mx[1, 1] = np.sum(x**2 * v4**2) * dx
    Mx[2, 2] = np.sum(x**2 * v8**2) * dx
    Mx[0, 1] = Mx[1, 0] = np.sum(x**2 * v0 * v4) * dx
    Mx[0, 2] = Mx[2, 0] = np.sum(x**2 * v0 * v8) * dx
    Mx[1, 2] = Mx[2, 1] = np.sum(x**2 * v4 * v8) * dx

    # Mf for freq^2 norm = c^T Mf c
    Mf = np.zeros((3, 3))
    Mf[0, 0] = np.sum(freqs**2 * np.abs(G0)**2) * dx
    Mf[1, 1] = np.sum(freqs**2 * np.abs(G4)**2) * dx
    Mf[2, 2] = np.sum(freqs**2 * np.abs(G8)**2) * dx
    Mf[0, 1] = Mf[1, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G4))) * dx
    Mf[0, 2] = Mf[2, 0] = np.sum(freqs**2 * np.real(G0 * np.conj(G8))) * dx
    Mf[1, 2] = Mf[2, 1] = np.sum(freqs**2 * np.real(G4 * np.conj(G8))) * dx

    def compute_bound_fast(coeffs):
        c = np.array(coeffs)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c

        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf

        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)

        return delta_x * delta_xi

    def objective(coeffs):
        return compute_bound_fast(coeffs)

    # Initial guess from prior work
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240

    best_coeffs = np.array([c0_init, c4_init, c8_init])
    best_bound = compute_bound_fast(best_coeffs)

    # Multiple local optimization attempts from different starting points
    # Using L-BFGS-B which is efficient for this smooth 3D problem
    start_points = [
        [c0_init, c4_init, c8_init],
        [-1.0, 0.04, 0.0003],
        [-1.2, 0.05, 0.0004],
        [-0.9, 0.03, 0.0002],
        [-1.1, 0.045, 0.00035],
        [-1.05, 0.035, 0.00025],
        [-1.15, 0.042, 0.00028],
        [-0.95, 0.038, 0.00032],
        [-1.08, 0.041, 0.00031],
        [-1.02, 0.039, 0.00029],
    ]

    for sp in start_points:
        if time.time() - start_time > deadline - 10:
            break
        try:
            res = minimize(
                objective,
                np.array(sp),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.success and np.isfinite(res.fun):
                if res.fun < best_bound:
                    best_bound = res.fun
                    best_coeffs = res.x.copy()
        except Exception:
            pass

    # Additional refinement with tighter tolerances from the best point found
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15}
            )
            if res.success and np.isfinite(res.fun):
                if res.fun < best_bound:
                    best_bound = res.fun
                    best_coeffs = res.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
