import numpy as np
from scipy.optimize import minimize
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < 0):
        return np.inf
    
    n = len(heights)
    w = 0.5 / n  # width of each interval, total support is 1/2
    
    # Sample the step function at fine resolution for FFT
    # Use a power of 2 for FFT efficiency
    M = 1 << 14  # 16384 points
    # The support is [-1/4, 1/4], total length 1/2
    # We need to cover the autoconvolution range [-1/2, 1/2]
    # Sample f on [-1/4, 1/4] with M points
    # Actually, let's use a different approach
    
    # Create a fine grid for the step function
    # The step function is constant on each of the N intervals
    # Let's sample at M points uniformly on [-1/4, 1/4]
    # Then use FFT to compute convolution
    
    # Better: use the fact that it's a step function
    # Sample at M points where M is multiple of N
    M = N * 32  # 19200 points, each step has 32 samples
    # Pad to power of 2
    M_fft = 1 << 15  # 32768
    
    # Create the fine-sampled function
    f_fine = np.zeros(M_fft)
    # Each of the N intervals corresponds to 32 fine points
    # The fine grid covers [-1/4, 1/4]
    for i in range(N):
        start = i * (M_fft // N)
        end = start + (M_fft // N)
        f_fine[start:end] = heights[i]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_fft)
    
    # The convolution is circular, but since f is zero-padded, it's correct
    # Find the maximum of the autoconvolution
    max_conv = np.max(conv)
    
    # Compute integral of f
    integral_f = np.sum(heights) * (0.5 / N)
    
    if integral_f == 0:
        return np.inf
    
    # Score = max_conv * support_length / (integral_f)^2
    # support_length = 1/2
    score = max_conv * 0.5 / (integral_f ** 2)
    
    return score

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Initial guess: raised cosine
    x = np.linspace(0, 1, N)
    h0 = 0.5 * (1 + np.cos(2 * np.pi * x))
    # Ensure nonnegative (it already is, min is 0)
    h0 = np.maximum(h0, 0)
    
    # Use scipy optimize
    result = None
    best_h = h0.copy()
    best_score = compute_score(h0)
    
    try:
        res = minimize(
            compute_score,
            h0,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 200, 'maxfun': 10000, 'maxls': 50}
        )
        if res.success or res.fun < best_score:
            best_h = res.x
            best_score = res.fun
    except Exception:
        pass
    
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return [max(0.0, float(v)) for v in best_h]
    
    # Try another optimization from a different start
    try:
        # Try a plateau function
        h1 = np.ones(N)
        # Make it slightly tapered
        taper = np.linspace(0.5, 1.0, N)
        h1 = taper
        res2 = minimize(
            compute_score,
            h1,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 100, 'maxfun': 5000, 'maxls': 50}
        )
        if res2.success or res2.fun < best_score:
            best_h = res2.x
            best_score = res2.fun
    except Exception:
        pass
    
    return [max(0.0, float(v)) for v in best_h]
