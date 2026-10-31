import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time

N = 600
NFFT = 2048  # next power of 2 >= 2*N - 1 = 1199

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    H = fft(h, n=NFFT)
    conv = np.real(ifft(H * H, n=NFFT))[:2 * N - 1]
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total == 0:
        return 0.0
    conv = _fft_convolve(h)
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Baseline: uniform
    best_h = np.ones(N)
    best_score = compute_autoconv_max(best_h)

    # x-grid for the 600 intervals on [-1/4, 1/4]
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False)

    # ---- Soft-max objective (smooth surrogate for the nonsmooth max) ----
    def soft_objective_direct(x, T=0.01):
        h = np.maximum(x, 0.0)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        conv = _fft_convolve(h)
        conv_max = np.max(conv)
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -100, 100)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)

    # ---- Gaussian mixture basis (structural idea from inspiration 1) ----
    n_gauss = 12

    def gaussians_to_h(params):
        h = np.zeros(N)
        for i in range(n_gauss):
            amp = np.exp(params[3 * i])
            center = params[3 * i + 1]
            width = np.exp(params[3 * i + 2])
            h += amp * np.exp(-0.5 * ((x_grid - center) / width) ** 2)
        return np.maximum(h, 0.0)

    def soft_objective_gauss(params, T=0.02):
        h = gaussians_to_h(params)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        conv = _fft_convolve(h)
        conv_max = np.max(conv)
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -100, 100)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)

    bounds_gauss = []
    for _ in range(n_gauss):
        bounds_gauss.append((-10, 5))    # log amplitude
        bounds_gauss.append((-0.25, 0.25))  # center
        bounds_gauss.append((-5, 2))     # log width

    np.random.seed(42)

    # ---- Stage 1: Differential evolution over Gaussian parameters (global) ----
    if time.time() < deadline - 70:
        try:
            de_result = optimize.differential_evolution(
                lambda p: soft_objective_gauss(p, T=0.05),
                bounds_gauss,
                maxiter=40,
                popsize=12,
                tol=1e-5,
                seed=42,
                polish=False,
                init='sobol'
            )
            h_de = gaussians_to_h(de_result.x)
            score_de = compute_autoconv_max(h_de)
            if score_de < best_score:
                best_score = score_de
                best_h = h_de.copy()
        except Exception:
            pass

    # ---- Stage 2: L-BFGS-B polish on Gaussian parameters ----
    if time.time() < deadline - 40:
        try:
            res_g = optimize.minimize(
                lambda p: soft_objective_gauss(p, T=0.01),
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds_gauss,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            h_g = gaussians_to_h(res_g.x)
            score_g = compute_autoconv_max(h_g)
            if score_g < best_score:
                best_score = score_g
                best_h = h_g.copy()
        except Exception:
            pass

    # ---- Stage 3: Direct L-BFGS-B on 600-dim space from multiple restarts ----
    restarts = [
        np.ones(N),
        np.maximum(1.0 + 0.15 * np.cos(2 * np.pi * np.linspace(0, 1, N, endpoint=False)), 0.01),
        np.maximum(1.0 + 0.15 * np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False)), 0.01),
        np.maximum(np.ones(N) + 0.10 * np.random.randn(N), 0.01),
        np.maximum(1.0 + 0.25 * np.abs(np.linspace(-1, 1, N)), 0.01),
        np.maximum(1.0 + 0.25 * np.exp(-5 * np.linspace(-1, 1, N) ** 2), 0.01),
        np.maximum(best_h, 0.001),
    ]

    for h0 in restarts:
        if time.time() > deadline - 20:
            break
        try:
            res = optimize.minimize(
                lambda x: soft_objective_direct(x, T=0.01),
                h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_new = np.maximum(res.x, 0.0)
            score_new = compute_autoconv_max(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
        except Exception:
            pass

    # ---- Stage 4: Fine polish with very low temperature ----
    if time.time() < deadline - 10:
        for _ in range(6):
            if time.time() > deadline - 5:
                break
            try:
                h_pert = best_h + 0.003 * np.random.randn(N)
                h_pert = np.maximum(h_pert, 0.0)
                res = optimize.minimize(
                    lambda x: soft_objective_direct(x, T=0.002),
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_new = np.maximum(res.x, 0.0)
                score_new = compute_autoconv_max(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
            except Exception:
                pass

    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
