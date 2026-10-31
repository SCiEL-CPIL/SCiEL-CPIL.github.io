"""Optimized step function for autoconvolution peak minimization."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 600
_INTERVAL = (0.25) / N  # 1/4 / 600

def _compute_score(heights):
    """Compute the normalized maximum of autoconvolution (lower is better)."""
    h = np.asarray(heights, dtype=np.float64)
    if h.size != N or np.any(h < -1e-12):
        return float('inf')
    h = np.maximum(h, 0.0)
    # Autoconvolution via FFT (zero-padded to avoid circular convolution)
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    # Normalize: divide by integral squared (or just take max since uniform scaling)
    # The "normalized maximum" - we normalize so that the integral is fixed
    # For a step function on equal intervals, integral = sum(h) * interval
    # Normalized autoconvolution peak = max(conv) / (sum(h)*interval)^2 * interval
    # Actually the standard normalization: (f*f)(x) / (||f||_1^2)
    # But since we're comparing, let's use max(conv) / (sum(h))^2 * N
    # Simpler: the score is max of normalized autoconvolution
    # Let's use: max(conv) / (sum(h) * interval)^2 * interval = max(conv) / (sum(h))^2 * N
    s = h.sum()
    if s <= 0:
        return float('inf')
    return float(np.max(conv) / (s * s) * N)

def _make_heights_from_params(params):
    """Convert continuous parameters to step heights."""
    # params: [sigma, cos_coef_1, cos_coef_2, ..., cos_coef_k]
    sigma = max(0.01, params[0])
    cos_coefs = params[1:]
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + _INTERVAL / 2
    # Gaussian envelope
    gauss = np.exp(-0.5 * (x / sigma) ** 2)
    # Even cosine polynomial
    poly = np.ones_like(x)
    for i, c in enumerate(cos_coefs):
        poly = poly + c * np.cos((i + 1) * np.pi * x / 0.25)
    heights = gauss * np.maximum(poly, 0.0)
    return heights

def _objective(params):
    heights = _make_heights_from_params(params)
    return _compute_score(heights)

def _gmm_objective(params):
    """GMM-based parameterization: [means, sigmas, weights] for K components."""
    K = 3
    means = params[0:K]
    sigmas = np.abs(params[K:2*K]) + 0.01
    weights = np.exp(params[2*K:3*K])
    weights = weights / weights.sum()
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + _INTERVAL / 2
    h = np.zeros_like(x)
    for m, s, w in zip(means, sigmas, weights):
        h += w * np.exp(-0.5 * ((x - m) / s) ** 2)
    h = np.maximum(h, 0.0)
    return _compute_score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial best: uniform
    best_heights = np.ones(N)
    best_score = _compute_score(best_heights)
    
    # Strategy 1: Gaussian × cosine polynomial with gradient descent
    n_cos = 4
    dim = 1 + n_cos
    x0 = np.concatenate([[0.08], [0.0] * n_cos])
    
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
        if time.time() < deadline:
            s = _compute_score(_make_heights_from_params(result.x))
            if s < best_score:
                best_score = s
                best_heights = _make_heights_from_params(result.x)
    except Exception:
        pass
    
    # Strategy 2: Try different initial sigma values
    if time.time() < deadline:
        for sigma_init in [0.05, 0.10, 0.12, 0.15]:
            if time.time() > deadline:
                break
            x0 = np.concatenate([[sigma_init], [0.0] * n_cos])
            try:
                result = minimize(_objective, x0, method='L-BFGS-B',
                                 options={'maxiter': 100, 'gtol': 1e-6})
                s = _compute_score(_make_heights_from_params(result.x))
                if s < best_score:
                    best_score = s
                    best_heights = _make_heights_from_params(result.x)
            except Exception:
                pass
    
    # Strategy 3: GMM with differential evolution
    if time.time() < deadline:
        K = 3
        bounds = [(-0.2, 0.2)] * K + [(0.01, 0.2)] * K + [(-3.0, 3.0)] * K
        try:
            de_result = differential_evolution(_gmm_objective, bounds, maxiter=50,
                                               popsize=20, tol=1e-5, seed=42,
                                               polish=True, workers=1)
            if time.time() < deadline:
                # Reconstruct heights from best GMM params
                params = de_result.x
                means = params[0:K]
                sigmas = np.abs(params[K:2*K]) + 0.01
                weights = np.exp(params[2*K:3*K])
                weights = weights / weights.sum()
                x = np.linspace(-0.25, 0.25, N, endpoint=False) + _INTERVAL / 2
                h = np.zeros_like(x)
                for m, s, w in zip(means, sigmas, weights):
                    h += w * np.exp(-0.5 * ((x - m) / s) ** 2)
                h = np.maximum(h, 0.0)
                s = _compute_score(h)
                if s < best_score:
                    best_score = s
                    best_heights = h
        except Exception:
            pass
    
    # Strategy 4: Local refinement of best heights via coordinate descent
    if time.time() < deadline:
        h = best_heights.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                for delta in [-0.5, -0.1, 0.1, 0.5, -0.01, 0.01]:
                    h_new = h.copy()
                    h_new[i] = max(0.0, h[i] + delta)
                    s = _compute_score(h_new)
                    if s < best_score - 1e-10:
                        h = h_new
                        best_score = s
                        improved = True
                        break
    
    # Ensure nonnegative and return as list
    best_heights = np.maximum(best_heights, 0.0)
    return list(best_heights)
