"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to improve an autoconvolution upper bound (C3).
    
    The strategy is to model the signed step function as a discretized band-limited
    signal by generating a smooth, oscillating height profile using a sum of
    low-frequency sine waves (Kaiser-Bessel window approximation). This leverages
    spectral properties of smooth functions to minimize the autoconvolution tail.
    """
    deadline = time.time() + 140  # 140s deadline guard (10s safety margin)
    
    # Generate a smooth, band-limited signal using a Kaiser-Bessel window approach
    # The Kaiser window has good sidelobe suppression properties
    n = N
    t = np.linspace(-1, 1, n)
    
    # Use a Kaiser-Bessel window with optimized beta parameter
    # Beta controls the trade-off between main lobe width and sidelobe level
    # Higher beta -> lower sidelobes but wider main lobe
    beta = 8.0
    
    # Kaiser window formula: I0(beta * sqrt(1 - x^2)) / I0(beta)
    # where I0 is the modified Bessel function of the first kind, order 0
    # We can approximate this or use scipy's kaiser function if available
    try:
        from scipy.signal.windows import kaiser
        window = kaiser(n, beta)
    except ImportError:
        # Fallback: approximate Kaiser window using numpy
        # I0(x) ≈ 1 + x^2/4 + x^4/64 + ... for small x
        # For our purposes, use a Gaussian approximation which has similar properties
        sigma = 0.5
        window = np.exp(-0.5 * (t / sigma) ** 2)
        window = window / window.max()
    
    # Create a signed step function: alternate signs with smooth envelope
    # The sign pattern helps with cancellation in autoconvolution
    signs = np.ones(n)
    # Create alternating sign pattern with smooth transitions
    # Use a smooth step function for transitions
    for i in range(1, n, 2):
        signs[i] = -1.0
    
    # Apply the window to create smooth, signed profile
    # The window tapers the edges, reducing discontinuities
    heights = signs * window
    
    # Normalize to have unit L2 norm (or some standard normalization)
    norm = np.linalg.norm(heights)
    if norm > 0:
        heights = heights / norm
    
    # Try to optimize the sign pattern using gradient-free optimization
    # Use a simple local search to improve the score
    best_heights = heights.copy()
    best_score = _compute_autoconvolution_bound(heights)
    
    # Local search: flip signs at random positions and keep if score improves
    rng = np.random.RandomState(42)
    n_iterations = 0
    max_iterations = 5000
    
    while time.time() < deadline and n_iterations < max_iterations:
        n_iterations += 1
        # Pick a random position to flip
        idx = rng.randint(n)
        trial = best_heights.copy()
        trial[idx] = -trial[idx]
        
        # Recompute score
        trial_score = _compute_autoconvolution_bound(trial)
        
        if trial_score < best_score:
            best_heights = trial
            best_score = trial_score
    
    return best_heights


def _compute_autoconvolution_bound(heights):
    """
    Compute the upper bound on C3 based on the autoconvolution of the step function.
    
    The autoconvolution upper bound is related to the L1 norm of the
    autoconvolution tail. For a step function with heights h[i],
    the autoconvolution is (h * h)[k] = sum_i h[i] h[i+k].
    
    We want to minimize the tail of this autoconvolution, which drives
    the C3 upper bound.
    """
    # Compute autoconvolution using FFT for efficiency
    n = len(heights)
    # Zero-pad to avoid circular convolution
    fft_size = 2 * n
    h_fft = np.fft.fft(heights, fft_size)
    autoconv = np.fft.ifft(h_fft * np.conj(h_fft)).real
    autoconv = autoconv[:n]
    
    # The C3 upper bound is related to the sum of |autoconv[k]| for k > 0
    # (excluding the peak at k=0)
    # A common measure is the L1 norm of the sidelobes
    if n > 1:
        sidelobes = np.abs(autoconv[1:])
        bound = np.sum(sidelobes)
    else:
        bound = 0.0
    
    # Also consider the ratio to the peak
    peak = abs(autoconv[0])
    if peak > 0:
        bound = bound / peak
    
    return bound
# EVOLVE-BLOCK-END
