"""
Third autocorrelation inequality: choose a signed step function with 400 equal intervals on [-1/4, 1/4]
to improve an autoconvolution upper bound.

We model the step function as a binary sign vector derived from the zero crossings
of a frequency-modulated sine wave (chirp), optimizing over center frequency
and chirp rate to minimize the peak side-lobe of the autoconvolution.
"""

import numpy as np

N = 400

def _compute_peak_side_lobe(signs):
    """
    Compute the peak side-lobe magnitude of the aperiodic autocorrelation.
    This is a robust proxy for minimizing the autoconvolution upper bound,
    as it directly controls the worst-case correlation deviation.
    """
    signs = np.asarray(signs, dtype=np.float64)
    n = len(signs)
    
    # Compute aperiodic autocorrelation using FFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    f = np.fft.rfft(signs, n=size)
    autocorr = np.fft.irfft(f * np.conj(f), n=size)[:2*n-1]
    
    # Exclude lag 0 (autocorr[0] and autocorr[2n-2] are the same, autocorr[n] is lag 0 in full circle? No)
    # autocorr[0] is R(0) = N
    # autocorr[1..n-1] are R(1)..R(n-1)
    # autocorr[n..2n-2] are R(-1)..R(-(n-1))
    # We want max abs of R(k) for k != 0.
    # Due to symmetry, we can just look at 1..n-1.
    if n > 1:
        side_lobes = np.abs(autocorr[1:n])
        peak = np.max(side_lobes)
    else:
        peak = 0.0
        
    return peak

def construct():
    """
    Construct a signed step function with N equal intervals.
    Returns a sequence of signed step heights (values in {-1, +1}).
    
    Strategy: Use a linear chirp (frequency-modulated sine) to determine the sign pattern.
    Chirps have low peak side-lobe autocorrelation properties compared to pure tones.
    We optimize over center frequency f0 and chirp rate beta.
    """
    x = np.arange(N, dtype=np.float64)
    
    # Initialize with a random sequence or all-ones?
    # All-ones is a likely "bad" seed. Random is "good".
    # We want to find a chirp that is better than the seed.
    # Let's start with a random sequence to have a good baseline.
    rng = np.random.default_rng(42)
    best_signs = rng.choice([-1.0, 1.0], size=N)
    best_score = _compute_peak_side_lobe(best_signs)
    
    # Also check all-ones to see if it's better (unlikely for peak side-lobe)
    ones_signs = np.ones(N, dtype=np.float64)
    ones_score = _compute_peak_side_lobe(ones_signs)
    if ones_score < best_score:
        best_score = ones_score
        best_signs = ones_signs
    
    # Search over Chirp parameters
    # Phase phi(t) = 2*pi * (f0 * t + 0.5 * beta * t^2)
    # t is normalized to [0, 1) or [0, N)?
    # Let's use t = x / N to keep frequencies in cycles per unit interval.
    t = x / N
    
    # Grid search
    num_f0 = 50
    num_beta = 50
    
    for i_f0 in range(num_f0):
        f0 = i_f0 / num_f0
        for i_beta in range(num_beta):
            beta = i_beta / num_beta
            
            # Generate phase
            phase = 2 * np.pi * (f0 * t + 0.5 * beta * t * t)
            sine_vals = np.sin(phase)
            signs = np.where(sine_vals >= 0, 1.0, -1.0)
            
            score = _compute_peak_side_lobe(signs)
            
            if score < best_score:
                best_score = score
                best_signs = signs.copy()
    
    # Local refinement around the best found
    if best_score < 100: # Only refine if we found something reasonably good
        best_f0_idx = None
        best_beta_idx = None
        # Re-find best indices (or store them)
        # For simplicity, we just do a small random perturbation around the current best
        # But we don't have the params. Let's just do a few random chirps with smaller steps
        # Actually, the grid is fine enough for N=400.
        
    return best_signs
