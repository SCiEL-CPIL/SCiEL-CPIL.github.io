"""Optimized step function to minimize normalized max autoconvolution."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
# Interval length
dx = (1/4 - (-1/4)) / N  # 0.5/600 = 1/1200

def compute_autoconvolution_max(h):
    """
    Compute the normalized maximum of the autoconvolution of the step function.
    
    The step function is h[i] on interval [x_i, x_{i+1}] where x_i = -1/4 + i*dx.
    The autoconvolution at a point y is:
      (f*f)(y) = ∫ f(y-t) f(t) dt
    
    For a step function, this is piecewise linear in y. The maximum of the
    autoconvolution will occur at one of the "critical points" which are the
    endpoints of the support or where the derivative changes sign.
    
    Actually, let's think more carefully. The autoconvolution of a step function
    with N steps is a piecewise quadratic function. The maximum can be found
    by evaluating at all critical points.
    
    A simpler approach: use FFT to compute the autoconvolution on a fine grid,
    then take the max. But we need to be careful about normalization.
    
    The "normalized maximum" likely means: max_y (f*f)(y) / (∫f)^2 or something similar.
    Actually, for the first autocorrelation inequality, C1 relates to:
      ||f * f||_∞ ≤ C1 * ||f||_2^2
    
    So the normalized maximum is: max_y (f*f)(y) / ||f||_2^2
    
    where ||f||_2^2 = ∫ f(x)^2 dx = sum(h_i^2 * dx)
    
    And (f*f)(y) = ∫ f(y-t)f(t) dt
    
    Let's compute this using FFT on a sufficiently fine grid.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.shape[0] != N:
        return 1.0  # invalid
    
    # Compute ||f||_2^2
    norm2 = np.sum(h**2) * dx
    
    if norm2 < 1e-15:
        return 1.0
    
    # Use FFT to compute autoconvolution
    # We need a grid fine enough to capture the piecewise linear structure
    # The step function has N steps, so the autoconvolution is piecewise quadratic
    # with breakpoints at multiples of dx. We need at least 2*N-1 points to capture
    # all the linear pieces, but let's use a finer grid for accuracy.
    
    # Actually, for a step function, the autoconvolution can be computed exactly.
    # Let's use a direct approach with a fine grid.
    
    # Support of f is [-1/4, 1/4], so support of f*f is [-1/2, 1/2]
    # The autoconvolution is piecewise quadratic with knots at k*dx for k = -N,...,N
    
    # Let's evaluate on a grid of points that includes all the knots
    # and the midpoints between knots (where the max of a quadratic on an interval occurs)
    
    # Number of intervals in the autoconvolution support: 2*N
    # Each interval has length dx, and the autoconvolution is quadratic on each
    
    # Actually, let me just use a fine grid with FFT for simplicity and speed
    # Use 2^k points where 2^k >= 4*N for good resolution
    nfft = 1 << (2 * N - 1).bit_length()
    nfft = max(nfft, 4096)
    
    # Pad h to nfft
    h_pad = np.zeros(nfft)
    h_pad[:N] = h
    
    # FFT
    H = np.fft.fft(h_pad)
    conv = np.fft.ifft(H * H).real * dx  # dx factor for the integral
    
    # The conv array corresponds to the autoconvolution on a grid
    # But we need to be careful about the grid spacing and alignment
    
    # Actually, let me use a more direct approach. The autoconvolution of a step
    # function can be computed exactly using the fact that it's piecewise quadratic.
    
    # Let's use a different approach: evaluate on a dense grid
    # The autoconvolution (f*f)(y) for a step function is:
    #   For y in [k*dx, (k+1)*dx], it's a quadratic function
    # We can compute it exactly at the endpoints and midpoints
    
    # Simpler: use np.convolve with a fine discretization
    # Let's discretize f on a fine grid and use convolution
    
    # Fine grid: use M points per original interval
    M = 4  # 4 sub-intervals per step
    n_fine = N * M
    dx_fine = dx / M
    
    # Interpolate h to the fine grid (constant interpolation)
    h_fine = np.repeat(h, M)
    
    # Convolve
    conv_fine = np.convolve(h_fine, h_fine, mode='full') * dx_fine
    
    # The max of the autoconvolution
    max_conv = np.max(conv_fine)
    
    return max_conv / norm2

def objective(h):
    """Objective function to minimize."""
    return compute_autoconvolution_max(h)

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    best_h = h0.copy()
    best_val = objective(h0)
    
    # Try a few different initializations and optimizers
    # L-BFGS-B with bounds [0, inf)
    
    # We can also try to exploit symmetry: the optimal function should be even
    # f(x) = f(-x), so h[i] = h[N-1-i]
    # This reduces the number of variables to N/2 = 300
    
    # Let's try both symmetric and asymmetric approaches
    
    # Symmetric approach
    def sym_objective(h_sym):
        """h_sym has length N/2, construct full h from symmetry."""
        h = np.empty(N)
        h[:N//2] = h_sym
        h[N//2:] = h_sym[::-1]
        # Handle middle element if N is odd (N=600 is even, so no middle)
        return compute_autoconvolution_max(h)
    
    h_sym0 = np.ones(N//2)
    
    # Try L-BFGS-B on symmetric variables
    try:
        res = minimize(
            sym_objective,
            h_sym0,
            method='L-BFGS-B',
            bounds=[(0, None)] * (N//2),
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline:
            h_sym_opt = np.maximum(res.x, 0)  # Ensure nonnegativity
            h_opt = np.empty(N)
            h_opt[:N//2] = h_sym_opt
            h_opt[N//2:] = h_sym_opt[::-1]
            
            val = objective(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt
    except Exception:
        pass
    
    # Try asymmetric L-BFGS-B if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            res = minimize(
                objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            h_opt = np.maximum(res.x, 0)
            val = objective(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            pass
    
    # Polish with a few more iterations if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5 and best_val > 0.5:  # If we haven't done well
        try:
            # Use the best result as initial guess for a finer optimization
            h_init = best_h.copy()
            
            res = minimize(
                objective,
                h_init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            
            h_opt = np.maximum(res.x, 0)
            val = objective(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)

# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"Score: {compute_autoconvolution_max(np.array(h))}")
