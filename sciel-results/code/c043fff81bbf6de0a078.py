import numpy as np
from scipy import optimize
import time

N = 600

def compute_autoconv_max_sidelobe(h):
    """Compute the normalized maximum of the autoconvolution sidelobes (k != 0)."""
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total == 0:
        return 0.0
    # Normalize h to sum to 1 for consistent scaling
    h_norm = h / total
    conv = np.convolve(h_norm, h_norm, 'full')
    # conv has length 2*N - 1
    # Peak is at index N-1
    # Sidelobes are everything else
    # We want the max of the sidelobes
    # Note: conv is symmetric
    peak_idx = N - 1
    # Create a mask to exclude the peak
    mask = np.ones(len(conv), dtype=bool)
    mask[peak_idx] = False
    # Also exclude the immediate neighbors if we want "far" sidelobes? 
    # No, "maximum" usually means global max.
    max_sidelobe = np.max(conv[mask])
    return max_sidelobe

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Initial guess: Gaussian (smooth)
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    best_h = np.exp(-((x - 0)**2) / (2 * 0.1**2))
    best_score = compute_autoconv_max_sidelobe(best_h)

    def gaussian_mixture_to_h(params, n_gaussians):
        h = np.zeros(N)
        for i in range(n_gaussians):
            center = params[3*i]
            width = max(abs(params[3*i+1]), 0.001)
            amp = max(params[3*i+2], 0)
            h += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
        return np.maximum(h, 0)

    def soft_max_sidelobe(h, T=0.01):
        h = np.asarray(h, dtype=float)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        h_norm = h / total
        conv = np.convolve(h_norm, h_norm, 'full')
        peak_idx = N - 1
        mask = np.ones(len(conv), dtype=bool)
        mask[peak_idx] = False
        sidelobes = conv[mask]
        pmax = np.max(sidelobes)
        if pmax < 1e-15:
            return 1e-10
        shifted = (sidelobes - pmax) / T
        shifted = np.clip(shifted, -50, 50)
        soft_max = pmax + T * np.log(np.sum(np.exp(shifted)))
        return soft_max

    # Optimize GMM parameters
    for n_gauss in [5, 10]:
        if time.time() > deadline - 20:
            break
        n_params = 3 * n_gauss
        np.random.seed(42)
        params = np.random.uniform(-0.2, 0.2, n_params)
        
        def objective(params):
            h = gaussian_mixture_to_h(params, n_gauss)
            return soft_max_sidelobe(h, T=0.01)

        try:
            result = optimize.minimize(
                objective, params, method='L-BFGS-B',
                options={'maxiter': 200}
            )
            h_opt = gaussian_mixture_to_h(result.x, n_gauss)
            score_opt = compute_autoconv_max_sidelobe(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Polish with L-BFGS-B on full vector
    if time.time() < deadline - 10:
        def soft_obj_direct(h):
            return soft_max_sidelobe(h, T=0.005)

        try:
            result = optimize.minimize(
                soft_obj_direct, best_h, method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100}
            )
            h_polished = np.maximum(result.x, 0)
            score_polished = compute_autoconv_max_sidelobe(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished.copy()
        except Exception:
            pass

    return list(best_h)
