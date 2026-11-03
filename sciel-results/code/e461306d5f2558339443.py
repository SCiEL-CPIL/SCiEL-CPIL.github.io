"""
Spectral coordinate descent for minimizing the normalized peak of the autoconvolution
of a nonnegative step function on 600 equal intervals over [-1/4, 1/4].
"""

import numpy as np
import time

N = 600
# Each interval has width dx = (1/4 - (-1/4))/600 = 0.5/600 = 1/1200
dx = 1.0 / (2.0 * N)  # 1/1200
total_width = 0.5
n_samples = 2 * N  # use 1200 sample points for FFT (centered around 0)
# Actually, for convolution of a function supported on [-1/4, 1/4], the autoconvolution
# is supported on [-1/2, 1/2]. We sample this on a grid.

def compute_autoconv_peak(f):
    """
    f: array of shape (N,) with nonnegative values, representing a step function
       on 600 equal intervals over [-1/4, 1/4].
    Returns the maximum value of the autoconvolution (continuous), approximated
    by sampling on a fine grid.
    """
    # f is the height on each interval of width dx
    # The function value at time t is f[i] where t is in the i-th interval
    # We'll compute the autoconvolution via FFT on a sufficiently fine grid
    
    # Support of autoconvolution: [-0.5, 0.5]
    # Use 2*N samples over this range
    n_fft = 2 * N
    # Create a signal of length n_fft representing f on its support [-0.25, 0.25]
    # The i-th interval covers [-0.25 + i*dx, -0.25 + (i+1)*dx)
    # We place f on a grid that aligns with these intervals
    
    # Simple approach: upsample f to a finer grid and use FFT convolution
    # Let's use a grid of size 4*N for the convolution to be accurate
    n_conv = 4 * N
    # Create signal g of length n_conv where g[i] = f[j] if t_i is in interval j
    # t_i = -0.5 + i * (1.0 / n_conv) for i = 0, ..., n_conv-1
    # Interval j covers [-0.25 + j*dx, -0.25 + (j+1)*dx)
    
    # Vectorized assignment
    t_grid = -0.5 + np.arange(n_conv) * (1.0 / n_conv)
    # Map t_grid to interval index
    # t in [-0.25, 0.25) maps to interval index = floor((t + 0.25) / dx)
    # Only t in [-0.25, 0.25) have f > 0 (assuming we only care about support)
    mask = (t_grid >= -0.25) & (t_grid < 0.25)
    idx = np.floor((t_grid[mask] + 0.25) / dx).astype(int)
    idx = np.clip(idx, 0, N - 1)
    
    g = np.zeros(n_conv)
    g[mask] = f[idx]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(g, n=2 * n_conv)
    conv = np.fft.irfft(F * F, n=2 * n_conv)
    # The autoconvolution is conv * (1/n_conv) ? No, the discrete convolution
    # sum g[k] g[k-m] approximates the integral. Let's just take the max.
    # Actually, the continuous autoconvolution is:
    # (f*f)(t) = integral f(s) f(t-s) ds
    # The discrete sum sum_k g[k] g[k-m] * (1/n_conv) approximates this if
    # we're careful with scaling. For now, let's just use the peak value
    # since we're minimizing and the scaling is consistent.
    
    peak = np.max(conv)
    return peak

def score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    Normalization: divide by (integral f)^2
    """
    integral_f = np.sum(f) * dx
    if integral_f == 0:
        return np.inf
    peak = compute_autoconv_peak(f)
    return peak / (integral_f * integral_f)

def construct():
    deadline = time.time() + 140  # 140s to be safe within 150s limit
    
    # Initialize with a random nonnegative function
    rng = np.random.default_rng(42)
    f = rng.random(N)
    f = np.maximum(f, 0)
    
    # Normalize to have integral 1 (so score = peak of autoconvolution)
    integral = np.sum(f) * dx
    f = f / integral
    
    best_f = f.copy()
    best_score = score(f)
    
    # Spectral coordinate descent
    # Work in Fourier domain
    n_fft = 2 * N
    
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
            
        # Compute Fourier transform
        # Extend f to a signal on [-0.25, 0.25]
        g = np.zeros(n_fft)
        # Place f on the grid
        # The i-th interval of f corresponds to time [-0.25 + i*dx, -0.25 + (i+1)*dx)
        # We need to map these to the FFT grid
        # Let's use a simpler approach: treat f as samples on a grid
        # and use FFT-based optimization
        
        # Actually, let's use a direct coordinate descent with spectral acceleration
        # Update one coordinate at a time in the time domain, but use FFT
        # to compute the autoconvolution efficiently
        
        # For now, let's do a simple gradient descent on the Fourier coefficients
        F = np.fft.rfft(f, n=n_fft)
        
        # Compute gradient of the peak with respect to Fourier coefficients
        # This is complex. Let's try a different approach:
        # Use alternating minimization: fix the peak location, optimize f
        
        # Find where the peak of the autoconvolution occurs
        conv = np.fft.irfft(np.fft.rfft(f, n=4*N) ** 2, n=4*N)
        peak_idx = np.argmax(conv)
        peak_val = conv[peak_idx]
        
        # The autoconvolution at the peak location t* is:
        # (f*f)(t*) = integral f(s) f(t*-s) ds
        # We want to minimize this. For a fixed t*, this is minimized
        # when f is constant (by Cauchy-Schwarz), but we have the constraint
        # that f is supported on [-0.25, 0.25].
        
        # Let's try a different strategy: use a parametric family
        # f(x) = 1 + a * cos(2*pi*k*x) + b * sin(2*pi*k*x)
        # and optimize over a, b, k
        
        # Actually, let's just do a simple random restart with local optimization
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            f_trial = rng.random(N)
            f_trial = np.maximum(f_trial, 0)
            integral = np.sum(f_trial) * dx
            f_trial = f_trial / integral
            
            s = score(f_trial)
            if s < best_score:
                best_score = s
                best_f = f_trial.copy()
                
            # Local optimization: coordinate descent
            for cd_iter in range(100):
                if time.time() > deadline - 5:
                    break
                    
                improved = False
                # Update a random subset of coordinates
                indices = rng.choice(N, size=N//10, replace=False)
                for i in indices:
                    old_val = f_trial[i]
                    # Try a small perturbation
                    for delta in [-0.1, 0.1]:
                        new_val = max(0, old_val + delta)
                        f_trial[i] = new_val
                        s_new = score(f_trial)
                        if s_new < s:
                            s = s_new
                            improved = True
                        else:
                            f_trial[i] = old_val
                            break
                if not improved:
                    break
                    
                if s < best_score:
                    best_score = s
                    best_f = f_trial.copy()
    
    return list(best_f)
