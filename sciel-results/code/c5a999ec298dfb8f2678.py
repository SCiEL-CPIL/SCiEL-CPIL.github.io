"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconvolution(h):
    """Compute the autoconvolution of a step function with N steps on [-1/4, 1/4].
    
    The step function is defined on N equal intervals covering [-1/4, 1/4].
    Each step has height h[i] on interval [x_i, x_{i+1}].
    
    The autoconvolution f*f(x) = integral f(t) f(x-t) dt.
    For piecewise constant functions, this can be computed exactly.
    
    Let dx = (1/2) / N = 1/(2N) be the width of each interval.
    The intervals are centered at positions c_i = -1/4 + (i + 0.5) * dx for i=0..N-1.
    
    The autoconvolution of two piecewise-constant functions can be computed by
    considering the overlap of intervals.
    """
    dx = 0.5 / N
    # The function is nonzero on [-1/4, 1/4]
    # The autoconvolution is nonzero on [-1/2, 1/2]
    
    # For piecewise constant functions, the autoconvolution at a point x
    # is the integral over t of f(t) f(x-t) dt.
    # f(t) = h[i] for t in [x_i, x_{i+1}] where x_i = -1/4 + i*dx
    # f(x-t) = h[j] for x-t in [x_j, x_{j+1}] i.e., t in [x - x_{j+1}, x - x_j]
    
    # The autoconvolution is a piecewise linear function (trapezoid-like pieces).
    # To get the supremum norm, we need to find the maximum value.
    
    # A more efficient approach: use FFT-based convolution with fine grid,
    # but for exactness with step functions, let's use the analytical structure.
    
    # Actually, for step functions, the autoconvolution is a continuous piecewise 
    # linear function. Its maximum occurs at one of the breakpoints or at critical points.
    # The breakpoints of the autoconvolution occur at x = c_i + c_j - 1/2 ... 
    # Actually, the breakpoints are at x = (x_i + x_j) - something...
    
    # Let me think more carefully. The autoconvolution (f*f)(x) = ∫ f(t) f(x-t) dt.
    # f is supported on [-1/4, 1/4]. So f*f is supported on [-1/2, 1/2].
    
    # For step functions, (f*f)(x) is piecewise linear. The breakpoints occur at
    # x = a_i + a_j where a_i are the endpoints of the step intervals.
    # Endpoints: x_i = -1/4 + i*dx for i=0..N.
    # So breakpoints are at x_i + x_j for all i,j, which gives values from -1/2 to 1/2.
    # There are (N+1)^2 such points, but many coincide.
    
    # A simpler approach: discretize finely and use numerical convolution,
    # then find the max. But this may miss the exact peak.
    
    # Alternative: use the fact that for nonnegative step functions, the autoconvolution
    # is unimodal-ish? No, not necessarily.
    
    # Let's use a fine-grid FFT approach with enough resolution to capture the peak.
    # The step function has N steps. The autoconvolution is piecewise linear with at most
    # O(N^2) linear pieces. Using a grid of size ~4*N^2 should be sufficient.
    
    n_grid = 4 * N * N + 1  # Should be odd for nice FFT
    if n_grid % 2 == 0:
        n_grid += 1
    
    # Define the grid on [-1/2, 1/2]
    x_grid = np.linspace(-0.5, 0.5, n_grid)
    dx_grid = x_grid[1] - x_grid[0]
    
    # Sample the step function on this grid
    # f(t) = h[i] for t in [-1/4 + i*dx, -1/4 + (i+1)*dx]
    # For a grid point t, find which interval it falls in
    # t in [-1/4, 1/4], index i = floor((t + 1/4) / dx)
    
    # Clip to support
    t_support = (x_grid >= -0.25) & (x_grid <= 0.25)
    f_grid = np.zeros(n_grid)
    idx = np.floor((x_grid[t_support] + 0.25) / dx).astype(int)
    idx = np.clip(idx, 0, N - 1)
    f_grid[t_support] = h[idx]
    
    # Compute autoconvolution using FFT
    # (f*f)(x) = ∫ f(t) f(x-t) dt
    # Using FFT: convolve f with f (with appropriate scaling)
    # For discrete convolution: sum_k f[k] f[n-k] * dx_grid
    # But we need to be careful with the definition.
    
    # Actually, let's just use scipy.signal.fftconvolve or direct convolution
    from scipy.signal import fftconvolve
    f_conv = fftconvolve(f_grid, f_grid, mode='full') * dx_grid
    # The result has length 2*n_grid - 1, corresponding to x from -1 to 1
    # But we only care about [-0.5, 0.5]
    # The center of the convolution corresponds to x=0
    # f_conv[k] corresponds to x = (k - (n_grid - 1)) * dx_grid
    x_conv = np.arange(-(n_grid - 1), n_grid) * dx_grid
    # Find indices where x_conv is in [-0.5, 0.5]
    mask = (x_conv >= -0.5) & (x_conv <= 0.5)
    f_conv_support = f_conv[mask]
    x_conv_support = x_conv[mask]
    
    # The supremum norm is the max of the autoconvolution
    sup_norm = np.max(f_conv_support) if len(f_conv_support) > 0 else 0.0
    
    return sup_norm

def _objective(h):
    """Objective to minimize: -(squared L2 norm) / (L1 norm * sup norm of autoconvolution)
    
    We want to maximize ratio = ||h||_2^2 / (||h||_1 * ||f*f||_inf)
    So we minimize -ratio.
    """
    h = np.maximum(h, 0)  # Ensure nonnegativity
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    
    if l1 < 1e-12:
        return 0.0  # Degenerate case
    
    sup_norm = _autoconvolution(h)
    if sup_norm < 1e-12:
        return 0.0
    
    ratio = l2_sq / (l1 * sup_norm)
    return -ratio

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    best_h = np.ones(N)
    best_score = -_objective(best_h)
    
    # Initial guesses
    guesses = [
        np.ones(N),
        np.linspace(0, 1, N),
        np.linspace(1, 0, N),
        np.array([1.0 if i < N // 2 else 0.0 for i in range(N)]),
        np.array([0.0 if i < N // 2 else 1.0 for i in range(N)]),
        np.sin(np.pi * np.arange(N) / N) + 0.1,
        np.cos(np.pi * np.arange(N) / N) + 0.5,
        np.exp(-((np.arange(N) - N/2) ** 2) / (2 * (N/4) ** 2)),
    ]
    
    for h0 in guesses:
        if time.time() > deadline:
            break
        h0 = np.maximum(h0, 0)
        try:
            res = minimize(_objective, h0, method='SLSQP',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.success and -res.fun > best_score:
                best_score = -res.fun
                best_h = np.maximum(res.x, 0)
        except Exception:
            pass
    
    # Additional local refinement
    if time.time() < deadline:
        try:
            res = minimize(_objective, best_h, method='SLSQP',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 500, 'ftol': 1e-12})
            if res.success and -res.fun > best_score:
                best_score = -res.fun
                best_h = np.maximum(res.x, 0)
        except Exception:
            pass
    
    return list(best_h)
