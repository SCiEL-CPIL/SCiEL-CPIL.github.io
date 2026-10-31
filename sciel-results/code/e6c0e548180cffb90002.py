"""Program using low-rank Gaussian mixture basis with differential evolution optimization."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve

N = 600

# Precompute grid points for the step function on [-1/4, 1/4]
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

# Precompute fine grid for convolution to speed up objective function
N_FINE = 4096
t_fine = np.linspace(-0.5, 0.5, N_FINE)
dt = t_fine[1] - t_fine[0]
idx_fine = np.floor((t_fine + 0.25) / _dx).astype(int)
idx_fine = np.clip(idx_fine, 0, N - 1)
in_support = (t_fine >= -0.25) & (t_fine < 0.25)

def _gaussian_mixture_heights(params, n_basis):
    """
    params: array of length 3 * n_basis
            [c1, w1, a1, c2, w2, a2, ...]
    Returns: array of N nonnegative step heights.
    """
    heights = np.zeros(N)
    for i in range(n_basis):
        c = params[3 * i]
        w = params[3 * i + 1]
        a = params[3 * i + 2]
        # Gaussian: a * exp(-(x - c)^2 / (2 * w^2))
        w_safe = max(w, 1e-6)
        heights += a * np.exp(-(( _centers - c) ** 2) / (2.0 * w_safe ** 2))
    return np.maximum(heights, 0.0)

def _compute_score(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    Score = max(f*f) / ||f||_2^2
    """
    l2_sq = np.sum(heights ** 2) * _dx
    
    if l2_sq <= 1e-15:
        return 1e10  # Avoid division by zero
    
    # Evaluate the step function on the fine grid
    f_fine = np.zeros(N_FINE)
    f_fine[in_support] = heights[idx_fine[in_support]]
    
    # Compute autoconvolution on the fine grid
    conv_fine = fftconvolve(f_fine, f_fine, mode='full') * dt
    max_conv = np.max(conv_fine)
    
    return max_conv / l2_sq

def construct():
    n_basis = 8
    # Parameter bounds:
    # Centers c_i in [-0.25, 0.25]
    # Widths w_i in [0.01, 0.5]
    # Amplitudes a_i in [0, 5]
    
    bounds = []
    for _ in range(n_basis):
        bounds.append((-0.25, 0.25))  # center
        bounds.append((0.01, 0.5))     # width
        bounds.append((0.0, 5.0))      # amplitude
    
    def objective(params):
        heights = _gaussian_mixture_heights(params, n_basis)
        return _compute_score(heights)
    
    # Use differential evolution with tuned parameters
    # Smaller fine grid allows more iterations
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=300,
        tol=1e-8,
        popsize=15,
        init='sobol',
        polish=False,
        updating='immediate',
        workers=1
    )
    
    best_params = result.x
    
    # Local refinement using L-BFGS-B for better convergence
    try:
        res_local = minimize(
            objective,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if res_local.success and res_local.fun < result.fun:
            best_params = res_local.x
    except Exception:
        pass
    
    best_heights = _gaussian_mixture_heights(best_params, n_basis)
    
    # Ensure nonnegative
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
