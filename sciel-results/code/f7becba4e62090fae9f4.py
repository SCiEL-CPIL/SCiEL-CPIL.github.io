import numpy as np
from scipy import optimize
import time

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    autoconv = DELTA * conv
    return autoconv

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    autoconv = compute_autoconv(h)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    score = max_ac * 0.5 / (integral ** 2)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 / N  # midpoints

    # Initialize with uniform
    best_h = np.ones(N)
    best_score = compute_score(best_h)

    # Quick parametric sweep for a good starting point
    for a in np.linspace(-0.4, 0.4, 9):
        h_try = np.maximum(1.0 + a * np.cos(2 * np.pi * x), 0)
        s = compute_score(h_try)
        if s < best_score:
            best_score = s
            best_h = h_try.copy()

    # Strategy: Optimize in Fourier coefficient space (smooth, low-dimensional)
    # Use more modes than before for better representation
    n_modes = 12
    # Basis: cos(k*pi*t), sin(k*pi*t) for k=1..n_modes/2? 
    # Actually, let's use k=1..n_modes for frequency k.
    # Domain is [-1/4, 1/4], period 1/2.
    # Fundamental frequency is 2*pi / (1/2) = 4*pi.
    # So modes are exp(i * 4*pi * k * t).
    # Let's define basis functions.
    t = x
    freqs = np.arange(1, n_modes + 1)
    
    def fourier_to_h(coeffs):
        h = np.ones(N)
        for k, c in enumerate(coeffs):
            f = 4 * np.pi * (k + 1)
            h += c * np.cos(f * t)
        return np.maximum(h, 0)

    def objective_fourier(coeffs):
        h = fourier_to_h(coeffs)
        return compute_score(h)

    # Use Nelder-Mead in Fourier space first (robust to non-differentiability of max)
    # Then polish with L-BFGS-B in direct space if time permits.

    # Multi-start Nelder-Mead in Fourier space
    np.random.seed(42)
    best_fourier_params = np.zeros(n_modes)
    best_fourier_score = best_score
    best_h_fourier = best_h.copy()

    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline - 60:
            break
        if i == 0:
            x0 = np.zeros(n_modes)
        else:
            x0 = np.random.randn(n_modes) * 0.05
        
        res = optimize.minimize(objective_fourier, x0, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
        if res.fun < best_fourier_score:
            best_fourier_score = res.fun
            best_fourier_params = res.x
            h_candidate = fourier_to_h(res.x)
            best_h_fourier = h_candidate.copy()
            best_score = res.fun
            best_h = h_candidate.copy()

    # Refine the best Fourier parameters with L-BFGS-B
    if time.time() < deadline - 30:
        res_ref = optimize.minimize(objective_fourier, best_fourier_params, method='L-BFGS-B',
                                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
        if res_ref.fun < best_score:
            best_score = res_ref.fun
            best_h = fourier_to_h(res_ref.x).copy()

    # Polish in direct space using L-BFGS-B
    # This allows the solution to deviate from the Fourier basis slightly to fit the max better.
    if time.time() < deadline - 10:
        def objective_direct(h_vec):
            # Soft constraint for non-negativity to keep smoothness? 
            # L-BFGS-B handles bounds.
            return compute_score(h_vec)
        
        # We need to be careful with the max operation's non-differentiability.
        # However, L-BFGS-B with finite differences often works okay on piecewise smooth functions
        # if the optimum is not exactly on a kink.
        
        bounds = [(0, None)] * N
        try:
            res_direct = optimize.minimize(objective_direct, best_h, method='L-BFGS-B',
                                           bounds=bounds, options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-5})
            if res_direct.fun < best_score:
                best_score = res_direct.fun
                best_h = res_direct.x.copy()
        except Exception:
            pass

    # Final check and cleanup
    if time.time() > deadline - 5:
        return best_h.tolist()
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
