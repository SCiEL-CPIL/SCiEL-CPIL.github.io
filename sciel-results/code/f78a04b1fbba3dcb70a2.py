"""
Direct frequency-domain phase optimization for signed autocorrelation minimization.

Strategy:
1. Fix the magnitude spectrum |X(k)| to be constant (or a smooth taper) to maximize 
   the "spread" of energy in frequency, which generally helps reduce time-domain peaks.
2. Optimize the phase spectrum phi(k) using BFGS to minimize the maximum absolute 
   value of the autoconvolution divided by the squared integral.
3. The metric is scale-invariant, so we normalize the signal to have unit sum (or 
   handle the ratio directly).
4. Use a smooth approximation of the max function (soft-max) to allow gradient-based 
   optimization, then polish with exact evaluation.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

# We only need the first N points of the autoconvolution for the metric?
# The task says "maximum absolute autoconvolution". Usually, for length N signals,
# the autoconvolution has length 2N-1.
# The metric is max(|c|) / (sum(x))^2.
# Note: sum(x) is the DC component X(0).
# If we fix |X(k)|, we can control the sum by scaling, but the ratio is scale invariant.
# Let's normalize the signal such that sum(x) = 1. Then we just minimize max(|c|).
# However, sum(x) might be negative. We can force sum(x) > 0 by flipping signs if needed.

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _soft_max_abs(conv, beta=100.0):
    """
    Smooth approximation of max(|conv|).
    Uses log-sum-exp trick for stability.
    """
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    # Shift to avoid overflow
    shifted = abs_conv - max_val
    # log(sum(exp(beta * shifted))) / beta
    # For large beta, this approximates max
    # We add max_val back
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _phase_objective(phi, beta=100.0, mag_spec=None):
    """
    Objective function for phase optimization.
    phi: array of phases for rfft bins (length _FFT_SIZE//2 + 1)
    mag_spec: fixed magnitude spectrum
    """
    # Construct spectrum
    # DC component (k=0)
    # For real signals, X(0) must be real. So phase[0] should be 0 or pi.
    # We fix phase[0] = 0 to ensure positive sum if possible, or let it be optimized but constrained.
    # Actually, if we fix |X|, and optimize phase, the sum is X(0).
    # If we want sum(x) to be non-zero, we need X(0) != 0.
    # Let's fix |X(0)| to be 1 and phase(0) to be 0.
    # For other bins, |X(k)| is fixed.
    
    # Reconstruct X
    X = np.zeros(_FFT_SIZE // 2 + 1, dtype=complex)
    X[0] = mag_spec[0] + 0j  # Force DC to be real and positive for now
    
    # For k > 0, X(k) = mag_spec[k] * exp(1j * phi[k])
    # Note: phi[0] is ignored in this construction
    
    X[1:] = mag_spec[1:] * np.exp(1j * phi[1:])
    
    # Inverse FFT to get time domain x
    x = np.fft.irfft(X, _FFT_SIZE)[:N]
    
    # Calculate metric
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    
    # Use soft max for smoothness
    conv = np.fft.irfft(X * X, _FFT_SIZE)[:2 * N - 1]
    soft_max = _soft_max_abs(conv, beta)
    
    return soft_max / (s * s)

def _phase_gradient(phi, beta=100.0, mag_spec=None):
    """
    Numerical gradient for phase optimization.
    """
    grad = np.zeros_like(phi)
    eps = 1e-6
    f0 = _phase_objective(phi, beta, mag_spec)
    
    for i in range(1, len(phi)):  # Skip DC phase
        phi_plus = phi.copy()
        phi_plus[i] += eps
        f_plus = _phase_objective(phi_plus, beta, mag_spec)
        
        phi_minus = phi.copy()
        phi_minus[i] -= eps
        f_minus = _phase_objective(phi_minus, beta, mag_spec)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
        
    return grad

def _try_phase_optimization(deadline, rng):
    """
    Optimize the phase spectrum using BFGS.
    """
    # Define magnitude spectrum
    # A flat spectrum (white noise) is a good start, but might have high peaks.
    # A tapered spectrum (e.g., raised cosine) might reduce time-domain sidelobes.
    # Let's try a flat spectrum first, as it's the "most spread out" in frequency.
    # Actually, for minimizing max autocorrelation, we want the autocorrelation to be as flat as possible.
    # This corresponds to a flat power spectrum in frequency? No, autocorrelation is the inverse FFT of power spectrum.
    # If Power Spectrum is flat, Autocorrelation is a delta function (at lag 0).
    # But we have a finite length signal.
    # If we make |X(k)| constant, the autocorrelation will have a peak at lag 0 and smaller values elsewhere.
    # The goal is to minimize the max absolute value. The peak at lag 0 is sum(x^2) which is fixed by Parseval if |X| is fixed.
    # Wait, the metric is max(|c|) / (sum(x))^2.
    # c[0] = sum(x^2).
    # If we fix |X(k)|, sum(x^2) is fixed.
    # sum(x) = X(0). If we fix |X(0)|, sum(x) is fixed.
    # So we are minimizing max(|c|) where c is the autoconvolution.
    # Since c[0] is fixed, we want to minimize the other peaks.
    
    k = np.arange(_FFT_SIZE // 2 + 1)
    # Flat magnitude
    mag_spec = np.ones(_FFT_SIZE // 2 + 1)
    # Maybe taper the edges to reduce ringing?
    # Let's try a Hann window on the magnitude
    # mag_spec = np.hanning(_FFT_SIZE // 2 + 1)
    # mag_spec[0] = 1.0 # Keep DC strong
    
    # Initial phase: random
    phi_init = rng.uniform(-np.pi, np.pi, size=_FFT_SIZE // 2 + 1)
    phi_init[0] = 0.0
    
    # Use BFGS
    # We need to be careful with the time limit.
    # BFGS can be slow if gradient is noisy.
    # Let's use a moderate beta for soft-max to start, then increase it.
    
    best_phi = phi_init.copy()
    best_val = _phase_objective(best_phi, 50.0, mag_spec)
    
    # Try a few different beta values
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 10:
            break
        
        # Reset or continue from previous? Let's start fresh for each beta to avoid local minima traps from previous beta.
        # Or continue? Continuing might be better.
        # Let's try continuing.
        
        res = minimize(
            _phase_objective,
            best_phi,
            args=(beta, mag_spec),
            method='BFGS',
            jac=_phase_gradient,
            options={'maxiter': 50, 'disp': False, 'gtol': 1e-5}
        )
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_phi = res.x
    
    # Convert best phase back to time domain signal
    X = np.zeros(_FFT_SIZE // 2 + 1, dtype=complex)
    X[0] = mag_spec[0] + 0j
    X[1:] = mag_spec[1:] * np.exp(1j * best_phi[1:])
    
    x = np.fft.irfft(X, _FFT_SIZE)[:N]
    
    # Normalize to sum=1
    s = np.sum(x)
    if abs(s) < 1e-12:
        return None, float('inf')
    
    x = x / s
    
    # Evaluate exact metric
    val = _metric(x)
    
    return x, val

def _try_local_search_fast(x, deadline, rng, max_rounds=50000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 100:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    
    rng = np.random.default_rng(2024)
    
    # Initial guess: flat signal
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Phase 1: Frequency Domain Phase Optimization
    if time.time() < deadline - 20:
        x_phase, val_phase = _try_phase_optimization(deadline - 20, rng)
        if x_phase is not None and val_phase < best_val:
            best_x = x_phase
            best_val = val_phase
    
    # Phase 2: Local Search in Time Domain
    if time.time() < deadline - 10:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 10, rng, max_rounds=50000)
    
    return best_x
