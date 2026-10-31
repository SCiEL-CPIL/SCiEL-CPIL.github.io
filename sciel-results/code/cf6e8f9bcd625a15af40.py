import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time

N = 600

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*n-1])
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = sum(h)
    if total < 1e-10:
        return 0.0
    return max(conv) / (total * total)

def _gaussian_mixture_to_step(params, N, x_min=-0.25, x_max=0.25):
    """Convert Gaussian mixture parameters to step function on [x_min, x_max].
    
    params: array of shape (K, 3) where each row is [weight, center, width]
    Returns: array of N nonnegative step heights
    """
    K = len(params)
    if K == 0:
        return np.ones(N)
    
    # Grid points for the 600 equal intervals
    # Each interval has width dx
    dx = (x_max - x_min) / N
    # Center of each interval
    centers = x_min + (np.arange(N) + 0.5) * dx
    
    h = np.zeros(N)
    for k in range(K):
        w, c, sigma = params[k]
        if w <= 0 or sigma <= 0:
            continue
        # Gaussian evaluated at interval centers
        g = w * np.exp(-0.5 * ((centers - c) / sigma) ** 2)
        h += g
    
    return h

def _step_to_gaussian_params(h, N, n_gaussians=5, x_min=-0.25, x_max=0.25):
    """Initialize Gaussian mixture parameters from a step function.
    
    Uses a simple approach: place gaussians at peaks of the function.
    """
    dx = (x_max - x_min) / N
    centers_grid = x_min + (np.arange(N) + 0.5) * dx
    
    # Find local maxima
    h_smooth = h.copy()
    if len(h_smooth) > 3:
        # Simple smoothing
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_smooth, kernel, mode='same')
    
    # Find peaks
    peaks = []
    for i in range(1, N-1):
        if h_smooth[i] > h_smooth[i-1] and h_smooth[i] >= h_smooth[i+1]:
            peaks.append(i)
    
    # If not enough peaks, add some evenly spaced ones
    if len(peaks) < n_gaussians:
        # Add evenly spaced positions
        for i in range(n_gaussians):
            pos = int(i * N / n_gaussians)
            if pos not in peaks:
                peaks.append(pos)
    
    # Take top n_gaussians peaks by height
    peaks = sorted(peaks, key=lambda i: -h_smooth[i])[:n_gaussians]
    
    params = np.zeros((n_gaussians, 3))
    for k, p in enumerate(peaks):
        params[k, 0] = h[p]  # weight
        params[k, 1] = centers_grid[p]  # center
        params[k, 2] = dx * 10  # initial width
    
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Initial guess: all ones
    h = np.ones(N)
    best_score = compute_autoconv_max(h)
    best_h = h.copy()
    
    # Number of Gaussians in the mixture
    n_gaussians = 8
    n_params = n_gaussians * 3  # weight, center, width per Gaussian
    
    # Bounds for parameters
    # weight: [0, 10]
    # center: [-0.25, 0.25]
    # width: [0.001, 0.1]
    bounds = []
    for _ in range(n_gaussians):
        bounds.append((0, 10))      # weight
        bounds.append((-0.25, 0.25)) # center
        bounds.append((0.001, 0.1))  # width
    
    def objective(params):
        """Objective function for differential evolution."""
        params = np.asarray(params).reshape(n_gaussians, 3)
        h = _gaussian_mixture_to_step(params, N)
        return compute_autoconv_max(h)
    
    # Initial population for differential evolution
    np.random.seed(42)
    
    # Create diverse initial population
    pop_size = 50
    x0 = np.zeros((pop_size, n_params))
    
    for i in range(pop_size):
        for k in range(n_gaussians):
            # Weight
            x0[i, k*3] = np.random.uniform(0.1, 1.0)
            # Center - spread across the domain
            x0[i, k*3+1] = -0.25 + (k + np.random.uniform(0, 0.5)) * 0.5 / n_gaussians
            # Width - vary the widths
            x0[i, k*3+2] = np.random.uniform(0.01, 0.05)
    
    # Run differential evolution
    try:
        result = optimize.differential_evolution(
            objective,
            bounds=bounds,
            x0=x0,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            init='sobol'
        )
        
        if result.success or result.fun < best_score:
            params_opt = result.x.reshape(n_gaussians, 3)
            h_opt = _gaussian_mixture_to_step(params_opt, N)
            score_opt = compute_autoconv_max(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
    except Exception as e:
        pass
    
    # If time remains, do a local refinement using L-BFGS on the step function
    if time.time() < deadline - 10:
        # Soft-max objective for local refinement
        def soft_objective(x, T):
            h = np.maximum(x, 0)
            conv = _fft_convolve(h)
            total = sum(h)
            if total < 1e-10:
                return 1e10
            conv_max = np.max(conv)
            shifted = (conv - conv_max) / T
            shifted = np.clip(shifted, -100, 100)
            soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
            return soft_max / (total * total)
        
        # Try a few different temperatures
        for T in [0.01, 0.005, 0.001]:
            if time.time() > deadline - 5:
                break
            try:
                res = optimize.minimize(
                    lambda x: soft_objective(x, T), best_h,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_refined = np.maximum(res.x, 0)
                score_refined = compute_autoconv_max(h_refined)
                if score_refined < best_score:
                    best_score = score_refined
                    best_h = h_refined.copy()
            except Exception:
                pass
    
    return list(best_h)
