"""Program using low-rank Gaussian mixture basis with differential evolution optimization."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve

N = 600

# Precompute grid points for the step function on [-1/4, 1/4]
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

def _gaussian_mixture_heights(params, n_basis=6):
    """
    params: array of length 3 * n_basis
            [c1, w1, a1, c2, w2, a2, ...]
    Returns: array of N nonnegative step heights.
    """
    n = n_basis
    heights = np.zeros(N)
    for i in range(n):
        c = params[3 * i]
        w = params[3 * i + 1]
        a = params[3 * i + 2]
        # Gaussian: a * exp(-(x - c)^2 / (2 * w^2))
        heights += a * np.exp(-(( _centers - c) ** 2) / (2.0 * w ** 2))
    return np.maximum(heights, 0.0)

def _compute_score(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    Score = max(f*f) / ||f||_2^2
    """
    # Compute L2 norm squared
    l2_sq = np.sum(heights ** 2) * _dx
    
    if l2_sq <= 1e-15:
        return 1e10  # Avoid division by zero
    
    # Compute autoconvolution on a fine grid
    # Increased resolution to reduce discretization error
    n_fine = 16384
    t_fine = np.linspace(-0.5, 0.5, n_fine)
    dt = t_fine[1] - t_fine[0]
    
    # Evaluate the step function on the fine grid
    x_fine = t_fine
    idx = np.floor((x_fine + 0.25) / _dx).astype(int)
    idx = np.clip(idx, 0, N - 1)
    
    in_support = (x_fine >= -0.25) & (x_fine < 0.25)
    f_fine = np.zeros(n_fine)
    f_fine[in_support] = heights[idx[in_support]]
    
    # Compute autoconvolution on the fine grid
    conv_fine = fftconvolve(f_fine, f_fine, mode='full') * dt
    max_conv = np.max(conv_fine)
    
    return max_conv / l2_sq

def construct():
    n_basis = 20
    # Parameter bounds:
    # Centers c_i in [-0.25, 0.25]
    # Widths w_i in [0.005, 0.02] (Narrow to avoid overlap)
    # Amplitudes a_i in [0, 20] (High to balance area)
    
    bounds = []
    for _ in range(n_basis):
        bounds.append((-0.25, 0.25))  # center
        bounds.append((0.005, 0.02))  # width
        bounds.append((0.0, 20.0))    # amplitude
    
    def objective(params):
        heights = _gaussian_mixture_heights(params, n_basis)
        return _compute_score(heights)
    
    # Use differential evolution with tuned parameters
    # Increased n_basis requires careful tuning to stay within time limit
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=50,
        tol=1e-6,
        popsize=15,
        init='sobol',
        polish=True,
        updating='immediate',
        workers=1
    )
    
    best_params = result.x
    best_heights = _gaussian_mixture_heights(best_params, n_basis)
    
    # Ensure nonnegative
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
