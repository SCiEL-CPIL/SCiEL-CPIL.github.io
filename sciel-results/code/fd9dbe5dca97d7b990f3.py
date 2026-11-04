"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0

    dx = 1.0 / (2 * N)
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False) + dx / 2

    def compute_score(h):
        """Exact same formula as seed, but using FFT for speed."""
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(h, 0.0)
        l2_sq = dx * np.sum(h ** 2)
        if l2_sq < 1e-15:
            return 1e10
        # FFT-based convolution
        n_fft = 1 << (2 * N).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        autoconv = np.fft.irfft(F * F, n=n_fft) * dx ** 2
        # For full convolution of two length-N arrays, the result has length 2N-1
        # In the FFT result (length n_fft), the linear convolution is at indices 0..N-1 
        # and n_fft-N+1..n_fft-1 (wrapped). But for autoconv of nonneg, max is at center.
        # The peak of autoconv is at lag 0, which in the circular FFT result is at index 0.
        # Actually, np.fft.irfft gives the circular convolution. For two length-N arrays
        # padded to n_fft >= 2N-1, the circular conv equals the linear conv at indices 0..2N-2.
        # The autoconv peak is at index 0 (lag 0).
        return np.max(autoconv) / l2_sq

    def build_from_gaussians(params):
        """h(x) = sum a_j * exp(-(x-mu_j)^2/(2*sigma_j^2))"""
        K = 4
        h = np.zeros(N)
        for j in range(K):
            a = params[j * 3]
            mu = params[j * 3 + 1]
            sigma = params[j * 3 + 2]
            sigma = max(sigma, 1e-6)
            h += a * np.exp(-(x_grid - mu) ** 2 / (2 * sigma ** 2))
        return np.maximum(h, 0.0)

    def objective(params):
        if time.time() > deadline:
            return 1e10
        h = build_from_gaussians(params)
        return compute_score(h)

    best_score = float('inf')
    best_h = None
    best_params = None

    K = 4
    n_params = K * 3

    # Initial guesses
    guesses = [
        np.array([1.0, 0.0, 0.15] + [0.0, 0.0, 0.15] * 3),
        np.array([0.5, -0.05, 0.1, 0.5, 0.05, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1]),
        np.array([1.0, 0.0, 0.25] + [0.0, 0.0, 0.25] * 3),
        np.array([0.3, -0.1, 0.08, 0.3, 0.0, 0.08, 0.3, 0.1, 0.08, 0.0, 0.0, 0.08]),
    ]

    for guess in guesses:
        if time.time() > deadline:
            break
        h = build_from_gaussians(guess)
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            best_params = guess.copy()

    # Differential evolution
    if time.time() < deadline - 30:
        bounds = [(0.0, 2.0), (-0.25, 0.25), (0.01, 0.3)] * K
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=100,
                popsize=18,
                tol=0.001,
                seed=42,
                polish=True,
                mutation=(0.4, 1.0),
                recombination=0.7,
            )
            if result.fun < best_score:
                h = build_from_gaussians(result.x)
                s = compute_score(h)
                if s < best_score:
                    best_score = s
                    best_h = h.copy()
                    best_params = result.x.copy()
        except Exception:
            pass

    # Local polish
    if best_params is not None and time.time() < deadline - 15:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8},
            )
            if res.fun < best_score:
                h = build_from_gaussians(res.x)
                s = compute_score(h)
                if s < best_score:
                    best_score = s
                    best_h = h.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)

    return list(best_h)

# EVOLVE-BLOCK-END
