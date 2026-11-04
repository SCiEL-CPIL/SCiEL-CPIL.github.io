"""
Spectral Envelope Optimization for Signed Step Functions.

This program minimizes the ratio of the maximum absolute off-diagonal 
autoconvolution to the squared sum of the sequence.

Key Insight:
The metric is scale-invariant. We can fix the sum of the sequence to 1 
(DC component of FFT to 1). The problem then reduces to minimizing the 
maximum sidelobe of the autoconvolution.

Strategy:
1. Parameterize the sequence by the magnitudes of its DFT coefficients 
   (excluding DC). We fix the phases to zero to ensure the sequence is 
   real and even-symmetric, which simplifies the optimization landscape.
2. Optimize the log-magnitudes of the DFT coefficients using L-BFGS-B 
   to shape the spectral envelope. A flat spectral envelope tends to 
   minimize time-domain peak sidelobes.
3. Perform a final time-domain polish using gradient descent on the 
   soft-max objective to fine-tune the result.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_soft_max(x, temp=1.0):
    """Smooth approximation of the max absolute off-diagonal autoconvolution."""
    c = _autoconv(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 0.0
    shifted = (abs_c - max_c) / temp
    log_sum_exp = max_c / temp + np.log(np.sum(np.exp(shifted)))
    soft_max = temp * log_sum_exp
    return soft_max

def _spectral_objective(log_mags, temp=1.0):
    """
    Objective function for spectral optimization.
    log_mags: array of log-magnitudes for k=1..N/2-1.
    We fix X_0 = 1 (sum x = 1) and phases to 0.
    """
    # Reconstruct X
    # rfft returns N//2 + 1 elements.
    # X[0] is DC. X[1:N//2] are the ones we optimize. X[N//2] is Nyquist (real).
    n_half = N // 2
    X = np.zeros(n_half + 1, dtype=complex)
    X[0] = 1.0 + 0j  # Fix DC to 1
    
    # Assign optimized magnitudes to k=1..N//2-1
    # log_mags has length N//2 - 1
    if len(log_mags) > 0:
        mags = np.exp(log_mags)
        # Ensure mags are not too large to avoid numerical issues
        mags = np.clip(mags, 1e-6, 1e6)
        X[1:n_half] = mags
        
    # Nyquist bin (if N is even)
    # We can fix it to 0 or optimize it. Let's fix it to 0 for simplicity 
    # as it's a single point. Or we can include it in log_mags.
    # Let's include it in the optimization for completeness if N is even.
    # Actually, standard rfft handles the last bin specially.
    # Let's just use the first N//2-1 bins and set Nyquist to 0.
    # This assumes an even-symmetric sequence without a Nyquist component.
    
    # Reconstruct x
    x = np.fft.irfft(X, N)
    
    # The sum of x is guaranteed to be 1 (since X[0]=1).
    # So the metric is just the max off-diagonal autoconvolution.
    return _objective_soft_max(x, temp)

def _optimize_spectral(log_mags0, temp=1.0, max_iter=200):
    """
    Optimize the spectral magnitudes.
    """
    try:
        res = minimize(
            _spectral_objective,
            log_mags0,
            args=(temp,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        best_log_mags = res.x
        # Reconstruct x
        n_half = N // 2
        X = np.zeros(n_half + 1, dtype=complex)
        X[0] = 1.0 + 0j
        mags = np.exp(best_log_mags)
        mags = np.clip(mags, 1e-6, 1e6)
        X[1:n_half] = mags
        x = np.fft.irfft(X, N)
        val = _metric_offdiag(x)
        return x, val
    except Exception:
        return None, float('inf')

def _gradient_descent_polish(x0, max_iter=500, initial_lr=0.01, deadline=None):
    """
    Projected gradient descent for final polish in time domain.
    """
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    # Normalize sum to 1
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    fsize = 1
    while fsize < 2 * N:
        fsize *= 2
        
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        # Soft-max weights
        shifted = (abs_c - max_c) / temp
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        
        # Gradient of soft-max w.r.t. c[1:]
        grad_abs_c = exp_vals / sum_exp
        c_off = c[1:]
        sign_c = np.sign(c_off)
        grad_c = grad_abs_c * sign_c
        
        # Build full convolution gradient vector
        w = np.zeros(2 * N - 1)
        w[1:] = grad_c
        
        # Gradient w.r.t. x: grad_x[m] = sum_k w[k] * x[k-m] (cross-correlation)
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w, fsize)
        grad_x = np.fft.irfft(W * np.conj(X), fsize)[:N]
        
        s = np.sum(x)
        metric = soft_max / (s * s)
        
        # Gradient of metric
        grad_metric = grad_x / (s * s) - 2.0 * soft_max / (s ** 3)
        
        x_new = x - lr * grad_metric
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        val = _metric_offdiag(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.995
        temp *= 0.999
        
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Initial best
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # 1. Spectral Optimization
    # We optimize log-magnitudes for k=1..N/2-1
    n_opt = N // 2 - 1
    
    # Start with a flat spectrum (log-mags = 0 => mags = 1)
    # Or a random spectrum
    log_mags0 = np.zeros(n_opt)
    
    # Try a few different initializations
    inits = [
        np.zeros(n_opt),
        rng.normal(0, 0.5, n_opt),
        np.linspace(0, -2, n_opt), # Decaying spectrum
        np.linspace(-2, 0, n_opt), # Rising spectrum
    ]
    
    for init in inits:
        if time.time() > deadline - 60:
            break
            
        x_spec, val_spec = _optimize_spectral(init, temp=1.0, max_iter=100)
        if x_spec is not None and val_spec < best_val:
            best_val = val_spec
            best_x = x_spec.copy()
            
        # Refine with lower temperature
        if x_spec is not None:
            if time.time() > deadline - 30:
                break
            x_ref, val_ref = _optimize_spectral(init, temp=0.1, max_iter=100)
            if x_ref is not None and val_ref < best_val:
                best_val = val_ref
                best_x = x_ref.copy()

    # 2. Time-domain Polish
    if time.time() < deadline - 10:
        try:
            polished = _gradient_descent_polish(
                best_x,
                max_iter=1000,
                initial_lr=0.005,
                deadline=deadline - 5
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass
            
    return best_x
