"""
Structural change: Direct 600-bin optimization via log-sum-exp smoothed max-autoconvolution,
combined with low-dimensional Fourier parameterization for global exploration.

Key changes from parent:
- Primary optimization is now direct L-BFGS-B on the 600-bin height vector using a
  log-sum-exp smoothed surrogate of the max autoconvolution, allowing the shape to
  escape the low-frequency Fourier manifold.
- Fourier parameterization (K=20 modes) is retained for diverse global restarts,
  providing structured initializations that are then refined in the full 600-dim space.
- A fast coarse-grid surrogate (using np.convolve on the 600 bins) is used for the
  inner optimization loop to allow many more iterations within the time budget, with
  final verification on the fine grid.
- Adaptive step-size coordinate descent polish at the end for local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0  # width of each interval

# Fine grid for accurate autoconvolution
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Fourier basis for fast conversion (K=20 modes)
K_FOURIER = 20
I = np.arange(N)
COS = np.cos(2.0 * np.pi * I[None, :] * np.arange(1, K_FOURIER + 1)[:, None] / N)
SIN = np.sin(2.0 * np.pi * I[None, :] * np.arange(1, K_FOURIER + 1)[:, None] / N)


def fourier_to_heights(params):
    """Convert Fourier params [a0, a1, b1, ..., aK, bK] to nonnegative heights."""
    a0 = params[0]
    a_coeffs = params[1::2][:K_FOURIER]
    b_coeffs = params[2::2][:K_FOURIER]
    heights = a0 + a_coeffs @ COS + b_coeffs @ SIN
    return np.clip(heights, 0.0, None)


def heights_to_fourier(heights):
    """Compute Fourier coefficients of a height vector."""
    a0 = np.mean(heights)
    a_coeffs = np.zeros(K_FOURIER)
    b_coeffs = np.zeros(K_FOURIER)
    for k in range(1, K_FOURIER + 1):
        a_coeffs[k - 1] = (2.0 / N) * np.sum(heights * np.cos(2.0 * np.pi * k * I / N))
        b_coeffs[k - 1] = (2.0 / N) * np.sum(heights * np.sin(2.0 * np.pi * k * I / N))
    params = np.zeros(2 * K_FOURIER + 1)
    params[0] = a0
    params[1::2] = a_coeffs
    params[2::2] = b_coeffs
    return params


def ratio_from_h_fine(h):
    """Accurate score using fine-grid FFT autoconvolution."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    max_conv = np.max(conv)
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')

    return max_conv / integral_sq


def ratio_from_h_coarse(h, tau=0.005):
    """
    Fast surrogate using coarse-grid convolution and log-sum-exp smoothing.
    Used for the inner optimization loop.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    # Coarse convolution: (h*h) on the 600-bin grid, scaled by W
    conv = np.convolve(h, h, mode='full') * W

    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def smooth_objective_fourier(params, tau=0.005):
    """Smooth surrogate for Fourier parameters using coarse grid."""
    h = fourier_to_heights(params)
    return ratio_from_h_coarse(h, tau)


def smooth_objective_direct(h, tau=0.005):
    """Smooth surrogate for direct 600-bin heights using coarse grid."""
    return ratio_from_h_coarse(h, tau)


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h_fine(best_h)

    np.random.seed(42)

    # --- Phase 1: Fourier-based global exploration ---
    # Generate diverse initializations in Fourier parameter space
    inits = []

    # 1. Near-uniform with small perturbations
    for _ in range(4):
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = 1.0
        p0[1:] = np.random.normal(0, 0.1, size=2 * K_FOURIER)
        inits.append(p0)

    # 2. Moderate modulation
    for _ in range(4):
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = 1.0
        p0[1:] = np.random.normal(0, 0.3, size=2 * K_FOURIER)
        inits.append(p0)

    # 3. Dominant low-frequency modes
    for k_dom in [1, 2, 3, 4]:
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = 1.0
        p0[2 * k_dom - 1] = np.random.uniform(0.2, 0.6)
        p0[2 * k_dom] = np.random.uniform(0.2, 0.6)
        for k in range(1, K_FOURIER + 1):
            if k != k_dom:
                p0[2 * k - 1] = np.random.normal(0, 0.05)
                p0[2 * k] = np.random.normal(0, 0.05)
        inits.append(p0)

    # 4. Cosine-bell shaped (single peak)
    for amp in [0.2, 0.4, 0.6, 0.8]:
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = 1.0
        p0[1] = -amp
        inits.append(p0)

    # 5. Random
    for _ in range(4):
        p0 = np.zeros(2 * K_FOURIER + 1)
        p0[0] = np.random.uniform(0.5, 1.5)
        p0[1:] = np.random.normal(0, 0.4, size=2 * K_FOURIER)
        inits.append(p0)

    # Optimize from each initialization using L-BFGS-B on the Fourier smooth surrogate
    for idx, p0 in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
        try:
            res = minimize(
                smooth_objective_fourier,
                p0,
                args=(0.005,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
            )
            if np.isfinite(res.fun):
                h_candidate = fourier_to_heights(res.x)
                val = ratio_from_h_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # --- Phase 2: Direct 600-bin L-BFGS-B refinement ---
    # Use the best Fourier result as starting point for direct optimization
    if time.time() < deadline - 40.0:
        h_start = best_h.copy()
        if h_start.sum() > 0:
            h_start = h_start * (N / h_start.sum())

        bounds = [(0.0, None)] * N
        try:
            res = minimize(
                smooth_objective_direct,
                h_start,
                args=(0.005,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                h_candidate = np.clip(res.x, 0.0, None)
                val = ratio_from_h_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # --- Phase 3: Additional direct L-BFGS-B restarts from perturbed best ---
    if time.time() < deadline - 25.0:
        for _ in range(3):
            if time.time() > deadline - 20.0:
                break
            h_perturb = best_h + np.random.normal(0, 0.05, size=N)
            h_perturb = np.clip(h_perturb, 0.0, None)
            if h_perturb.sum() > 0:
                h_perturb = h_perturb * (N / h_perturb.sum())
            bounds = [(0.0, None)] * N
            try:
                res = minimize(
                    smooth_objective_direct,
                    h_perturb,
                    args=(0.005,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
                )
                if np.isfinite(res.fun):
                    h_candidate = np.clip(res.x, 0.0, None)
                    val = ratio_from_h_fine(h_candidate)
                    if val < best_val and np.isfinite(val):
                        best_val = val
                        best_h = h_candidate.copy()
            except Exception:
                pass

    # --- Phase 4: Coordinate descent polish ---
    if time.time() < deadline - 5.0:
        step_size = 0.03
        for it in range(3000):
            if time.time() > deadline - 1.0:
                break
            n_perturb = np.random.randint(3, 12)
            idx = np.random.choice(N, n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += np.random.randn(n_perturb) * step_size
            h_try = np.clip(h_try, 0.0, None)

            val = ratio_from_h_fine(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
                step_size = max(step_size * 0.95, 0.003)
            else:
                step_size = min(step_size * 1.05, 0.1)

    # --- Phase 5: Very fine polish ---
    if time.time() < deadline - 1.5:
        step_size = 0.005
        for it in range(500):
            if time.time() > deadline - 0.5:
                break
            idx = np.random.choice(N, 4, replace=False)
            h_try = best_h.copy()
            h_try[idx] += np.random.randn(4) * step_size
            h_try = np.clip(h_try, 0.0, None)
            val = ratio_from_h_fine(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()

    return list(best_h)
