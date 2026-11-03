"""
Optimized program for the First Autocorrelation Inequality task.

Approach:
1. Represent the step function by its first M Fourier cosine coefficients (symmetric, even function).
2. Use the true autoconvolution of the step function defined on [-1/4, 1/4] (zero outside).
3. Compute the autoconvolution via FFT for efficiency.
4. Optimize the Fourier coefficients to minimize the normalized maximum of the autoconvolution.
"""

import numpy as np
from scipy.optimize import minimize
import sys

N = 600
L = 0.5  # Length of the interval [-1/4, 1/4]

def construct():
    M = 15  # Number of harmonics
    
    # Discretization for computing f and its autoconvolution
    n_grid = 4096
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    dx = L / n_grid
    
    # Precompute cosine basis for the grid
    k_vals = np.arange(M + 1)
    cos_grid = np.cos(2 * np.pi * np.outer(x, k_vals) / L)  # shape (n_grid, M+1)
    
    def compute_f(a):
        """Compute f(x) for given coefficients a."""
        # f(x) = a[0] + sum_{k=1}^M a[k] * cos(2*pi*k*x/L)
        return a[0] + cos_grid[:, 1:] @ a[1:]
    
    def compute_autoconv_max(f_vals):
        """Compute the maximum of the true autoconvolution of f_vals (defined on [-L/2, L/2], zero outside)."""
        # The autoconvolution g(t) = integral f(tau) f(t-tau) dtau
        # f is supported on [-L/2, L/2], so g is supported on [-L, L]
        # We can compute this via FFT
        
        # Pad f to length 2*n_grid for linear convolution
        f_padded = np.zeros(2 * n_grid)
        f_padded[:n_grid] = f_vals
        
        # FFT-based convolution
        F = np.fft.fft(f_padded)
        G = F * F  # Autoconvolution in frequency domain
        
        # Inverse FFT
        g = np.fft.ifft(G).real
        
        # The convolution result is on a grid of length 2*n_grid
        # The original f was on [-L/2, L/2] with spacing dx
        # The convolution g(t) is on [-L, L] with spacing dx
        # The maximum of g is what we want
        return np.max(g)
    
    def compute_energy(f_vals):
        """Compute the L2 energy of f."""
        return np.sum(f_vals**2) * dx
    
    def objective(a):
        """Objective function: normalized max autoconvolution + penalty for negative f."""
        f_vals = compute_f(a)
        
        # Penalty for negative values
        neg_penalty = np.sum(np.maximum(0, -f_vals)**2)
        
        # Compute score
        energy = compute_energy(f_vals)
        if energy < 1e-10:
            return 1e10 + neg_penalty
        
        max_autoconv = compute_autoconv_max(f_vals)
        score = max_autoconv / energy
        
        return score + 100.0 * neg_penalty
    
    # Initial guess: uniform function (a[0] = 1, others 0)
    a0 = np.zeros(M + 1)
    a0[0] = 1.0
    
    # Run optimization with multiple restarts
    best_score = np.inf
    best_a = a0.copy()
    
    np.random.seed(42)
    for restart in range(5):
        if restart == 0:
            a = a0.copy()
        else:
            # Small perturbation around uniform
            a = a0 + 0.1 * np.random.randn(M + 1)
            a[0] = max(0.1, a[0])  # Keep DC component positive
        
        # Use Nelder-Mead for derivative-free optimization
        result = minimize(objective, a, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.fun < best_score:
            # Check if the solution is feasible (nonnegative)
            f_check = compute_f(result.x)
            if np.all(f_check >= -1e-4):
                best_score = result.fun
                best_a = result.x.copy()
    
    # Evaluate on the 600 intervals
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Compute f at centers
    cos_centers = np.cos(2 * np.pi * np.outer(centers, k_vals) / L)
    heights = best_a[0] + cos_centers[:, 1:] @ best_a[1:]
    
    # Ensure nonnegativity
    heights = np.maximum(0, heights)
    
    return list(heights)
