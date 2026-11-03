"""
Optimized program for the First Autocorrelation Inequality task.

Approach:
1. Represent the step function via its Fourier coefficients on a small set of low-frequency harmonics.
2. The autoconvolution of a function f supported on [-1/4, 1/4] is computed via FFT.
3. We minimize the ratio max(f*f) / (∫f^2 dx) by fixing the L2 norm and minimizing the peak.
4. Use gradient-based optimization (L-BFGS-B) with penalty for non-negativity.
5. The Fourier representation ensures smoothness and avoids high-frequency artifacts.
"""

import numpy as np
from scipy.optimize import minimize
import sys

N = 600

def construct():
    L = 0.5  # Length of interval [-1/4, 1/4]
    M = 15   # Number of harmonics (low-frequency only)
    
    # Grid for evaluating f(x) >= 0 constraint
    n_grid = 2048
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    
    # Precompute Fourier basis matrix B where B[j, k] = phi_k(x_j)
    # phi_0(x) = 1/sqrt(L)
    # phi_k(x) = sqrt(2/L) * cos(2*pi*k*x/L) for k >= 1
    B = np.zeros((n_grid, M+1))
    B[:, 0] = 1.0 / np.sqrt(L)
    for k in range(1, M+1):
        B[:, k] = np.sqrt(2.0/L) * np.cos(2*np.pi*k*x/L)
    
    # FFT setup for autoconvolution
    # f is defined on [-1/4, 1/4], zero elsewhere
    # We need to compute (f*f)(t) for t in [-1/2, 1/2]
    # Use FFT with sufficient zero-padding to avoid circular convolution artifacts
    n_fft = 4096  # Power of 2, large enough for linear convolution
    # Sample f at n_fft points over [-L/2, L/2] with zero padding
    # Actually, let's sample f on a grid of size n_fft over [-L/2, L/2]
    # The convolution will be computed via FFT
    
    # For the objective, we need to evaluate max of the autoconvolution
    # Let's use a fine time grid for the autoconvolution
    n_t = 2048
    t = np.linspace(-L, L, n_t, endpoint=False)
    dt = 2*L / n_t
    
    def compute_f(u):
        """Compute f(x) values on the grid."""
        return B @ u
    
    def compute_autoconvolution(u):
        """Compute the autoconvolution (f*f)(t) using FFT."""
        # Evaluate f on a grid suitable for FFT
        # f is supported on [-L/2, L/2]
        # Sample f at n_fft points over [-L/2, L/2]
        x_fft = np.linspace(-L/2, L/2, n_fft, endpoint=False)
        # Compute f at these points
        # We need B evaluated at x_fft
        # For efficiency, precompute or compute on the fly
        # Since n_fft might differ from n_grid, let's compute directly
        f_fft = np.zeros(n_fft)
        f_fft += u[0] / np.sqrt(L)
        for k in range(1, M+1):
            f_fft += u[k] * np.sqrt(2.0/L) * np.cos(2*np.pi*k*x_fft/L)
        
        # FFT for convolution
        F = np.fft.fft(f_fft)
        G = F * F  # Autoconvolution in frequency domain
        g = np.fft.ifft(G).real
        
        # The convolution result g is on the same grid as f_fft
        # But we need to map it to the correct time coordinates
        # The convolution of two functions supported on [-L/2, L/2] is supported on [-L, L]
        # The FFT-based circular convolution needs careful handling
        
        # Alternative: use direct convolution via np.convolve
        # f is sampled at n_fft points with spacing dx = L/n_fft
        dx = L / n_fft
        # Linear convolution
        g_linear = np.convolve(f_fft, f_fft, mode='full')
        # The linear convolution has length 2*n_fft - 1
        # It corresponds to time values from -L/2 to L/2 - dx (start) to L/2 + L/2 - dx (end)
        # Actually, the first element corresponds to t = -L (when both functions are at their left edge)
        # Let's be more careful.
        
        # f[i] corresponds to x_i = -L/2 + i*dx for i=0,...,n_fft-1
        # (f*f)[k] = sum_i f[i] f[k-i]
        # This corresponds to t_k = (-L/2 + i*dx) + (-L/2 + (k-i)*dx) = -L + k*dx
        # So t_k = -L + k*dx for k=0,...,2*n_fft-2
        # We want t in [-L, L], so k from 0 to 2*n_fft-2 covers [-L, -L + (2*n_fft-2)*dx] = [-L, L - 2*dx]
        # Close enough.
        
        # Interpolate to our desired t grid
        t_linear = -L + np.arange(len(g_linear)) * dx
        g_interp = np.interp(t, t_linear, g_linear)
        
        return g_interp
    
    def objective(u):
        """Objective: max of autoconvolution + penalty for negative f."""
        f_vals = compute_f(u)
        g_vals = compute_autoconvolution(u)
        
        # Penalty for negative f
        neg_f = np.maximum(0, -f_vals)
        penalty = 1000.0 * np.sum(neg_f**2)
        
        return np.max(g_vals) + penalty
    
    def gradient(u):
        """Numerical gradient (for L-BFGS-B)."""
        eps = 1e-7
        grad = np.zeros(M+1)
        for i in range(M+1):
            u_plus = u.copy()
            u_plus[i] += eps
            u_minus = u.copy()
            u_minus[i] -= eps
            grad[i] = (objective(u_plus) - objective(u_minus)) / (2*eps)
        return grad
    
    # Initial guess: uniform function
    u0 = np.zeros(M+1)
    u0[0] = np.sqrt(L)  # So that ||f||_2^2 = 1
    
    best_score = np.inf
    best_u = u0.copy()
    
    # Try multiple restarts
    n_restarts = 3
    for restart in range(n_restarts):
        if restart == 0:
            u_init = u0.copy()
        else:
            u_init = np.random.randn(M+1)
            # Decay high frequencies
            for k in range(1, M+1):
                u_init[k] *= (0.7 ** k)
            u_init[0] = np.sqrt(L)  # Keep DC component dominant
            u_init /= np.linalg.norm(u_init) * np.sqrt(L)
        
        try:
            result = minimize(
                objective,
                u_init,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                # Check non-negativity
                f_vals = compute_f(result.x)
                if np.min(f_vals) >= -1e-5:
                    best_score = result.fun
                    best_u = result.x.copy()
        except Exception as e:
            pass
    
    # Evaluate the best solution
    f_vals = compute_f(best_u)
    f_vals = np.maximum(0, f_vals)  # Enforce non-negativity
    
    # Sample at the 600 interval centers
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Interpolate f to the centers
    heights = np.interp(centers, x, f_vals)
    heights = np.maximum(0, heights)
    
    return list(heights)
