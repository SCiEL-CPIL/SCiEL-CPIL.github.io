"""
Third autocorrelation inequality: choose a signed step function with 400 equal intervals on [-1/4, 1/4]
to improve an autoconvolution upper bound.

We model the step function as a binary sign vector derived from the zero crossings
of a low-frequency sine wave, optimizing over frequency and phase to minimize
the autoconvolution upper bound C3.
"""

import numpy as np

N = 400

def _compute_c3(signs):
    """
    Compute the autoconvolution upper bound C3 for a given sign sequence.
    C3 is defined as the maximum over all lags of the absolute value of
    the aperiodic autocorrelation, normalized appropriately.
    
    For the third autocorrelation inequality, C3 relates to the peak
    side-lobe level of the autocorrelation. We use the L1 norm of the
    aperiodic autocorrelation as a proxy, which is a valid upper bound.
    """
    signs = np.asarray(signs, dtype=np.float64)
    n = len(signs)
    
    # Compute aperiodic autocorrelation using FFT for efficiency
    # Pad to next power of 2 for efficient FFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    f = np.fft.rfft(signs, n=size)
    autocorr = np.fft.irfft(f * np.conj(f), n=size)[:2*n-1]
    
    # The aperiodic autocorrelation R(k) for k=0,...,n-1
    # R(k) = sum_{i=0}^{n-1-k} signs[i] * signs[i+k]
    # autocorr[0] = R(0), autocorr[1] = R(1), ..., autocorr[n-1] = R(n-1)
    # autocorr[n] = R(-1), ..., autocorr[2n-2] = R(-(n-1))
    
    # L1 norm of full autocorrelation (all lags from -(n-1) to n-1)
    l1_norm = np.sum(np.abs(autocorr))
    
    return l1_norm

def construct():
    """
    Construct a signed step function with N equal intervals.
    Returns a sequence of signed step heights (values in {-1, +1}).
    
    Strategy: Use a low-frequency sine wave to determine the sign pattern.
    We optimize over frequency and phase to minimize the L1 norm of the
    aperiodic autoconvolution. We use a dense grid of frequencies and phases.
    """
    x = np.arange(N, dtype=np.float64)
    
    best_signs = np.ones(N, dtype=np.float64)
    best_score = _compute_c3(best_signs)
    
    # Search over frequencies k=1 to N//2 (higher frequencies alias)
    # and phases on a fine grid
    for k in range(1, N // 2 + 1):
        # For each frequency, try multiple phases
        # The sign pattern depends on sign(sin(2*pi*k*x/N + phi))
        # Due to the binary nature, phase shifts of 2*pi/k give the same pattern shifted
        # So we only need to consider phi in [0, 2*pi/k)
        # But to be safe, let's try a moderate number of phases
        
        # Number of distinct phase shifts that give different patterns
        # For a sine with frequency k over N points, the pattern repeats
        # after a shift of N/gcd(k,N) points, which corresponds to phase 2*pi*k*(N/gcd(k,N))/N = 2*pi*k/gcd(k,N)
        # So there are gcd(k,N) distinct phases
        
        g = np.gcd(k, N)
        num_phases = max(1, min(g, 20))  # Cap at 20 for speed
        
        for p in range(num_phases):
            phi = 2 * np.pi * p / num_phases
            sine_vals = np.sin(2 * np.pi * k * x / N + phi)
            signs = np.where(sine_vals >= 0, 1.0, -1.0)
            
            score = _compute_c3(signs)
            
            if score < best_score:
                best_score = score
                best_signs = signs.copy()
    
    return best_signs
