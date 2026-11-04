import numpy as np
import time

N = 50
W = 1.0 / 25.0

def _autoconv_max(h):
    """Compute the supremum norm of the autoconvolution of piecewise-constant h."""
    n = len(h)
    # Pad to avoid circular convolution
    n_pad = 2 * n + 1
    h_pad = np.zeros(n_pad)
    h_pad[:n] = h
    h_conv = np.fft.ifft(np.fft.fft(h_pad) * np.fft.fft(h_pad))
    h_conv = np.real(h_conv)
    # The autoconvolution at grid point k is h_conv[k] * W
    # The maximum of the piecewise-linear function occurs at a grid point (knot)
    return np.max(h_conv) * W

def _ratio(h):
    """Compute the ratio ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    l2_sq = np.sum(h**2) * W
    l1 = np.sum(h) * W
    autoconv_max = _autoconv_max(h)
    if l1 * autoconv_max < 1e-30:
        return 0.0
    return l2_sq / (l1 * autoconv_max)

def construct():
    # We use a two-lobe Gaussian model: h(x) = exp(-((x-delta)^2)/(2*sigma^2)) + exp(-((x+delta)^2)/(2*sigma^2))
    # x is the grid index centered at (N-1)/2.
    # We optimize delta (offset) and sigma (width).
    
    # Grid of indices centered at 0
    x = np.arange(N) - (N - 1) / 2.0
    
    best_h = None
    best_ratio = -1.0
    
    # Start time for safety
    start_time = time.time()
    time_limit = 100.0  # Keep well under 120s
    
    # Search parameters
    # delta: offset from center. Range 0 to ~10 (grid units)
    # sigma: width. Range 0.5 to ~5 (grid units)
    
    # Coarse to fine search
    # Phase 1: Coarse grid
    deltas = np.linspace(0.5, 10.0, 20)
    sigmas = np.linspace(0.5, 5.0, 20)
    
    for d in deltas:
        if time.time() - start_time > time_limit:
            break
        for s in sigmas:
            if time.time() - start_time > time_limit:
                break
            
            # Construct the shape
            # Note: We want non-negative heights. The sum of two Gaussians is positive.
            # We normalize the amplitude to 1.0 to avoid scaling issues.
            h_raw = np.exp(-((x - d)**2) / (2 * s**2)) + np.exp(-((x + d)**2) / (2 * s**2))
            
            # Normalize so that max is 1.0? Or sum is 1.0? 
            # The ratio is scale invariant, so we can just use h_raw directly.
            # However, to be safe against extreme values, we can normalize by max.
            h_candidate = h_raw / np.max(h_raw)
            
            r = _ratio(h_candidate)
            
            if r > best_ratio:
                best_ratio = r
                best_h = h_candidate.copy()
                
    # Phase 2: Fine grid around the best found
    if best_h is not None:
        # We need to track which d and s gave the best result to refine.
        # Since we didn't store them, let's just do a local refinement by perturbing the found h?
        # Better: Re-run the best region with finer steps.
        # For simplicity and robustness, let's just do a second pass with a slightly finer grid
        # centered around the previous best parameters if we had stored them.
        # Since we didn't store d,s, we will just do a dense grid over a smaller range
        # or accept the coarse result. Given the time constraint, a dense global grid
        # or a few iterations of local search is better.
        
        # Let's do a simple local search on the parameters by re-evaluating a fine grid
        # over the whole range but with more points, if time permits.
        # Actually, the coarse grid above is 20x20=400 evaluations. Each is ~fast (FFT of size 101).
        # 400 * 100us = 40ms. Very fast.
        # Let's do a finer grid: 50x50 = 2500 evaluations. Still fast.
        
        deltas_fine = np.linspace(0.5, 10.0, 50)
        sigmas_fine = np.linspace(0.5, 5.0, 50)
        
        for d in deltas_fine:
            if time.time() - start_time > time_limit:
                break
            for s in sigmas_fine:
                if time.time() - start_time > time_limit:
                    break
                
                h_raw = np.exp(-((x - d)**2) / (2 * s**2)) + np.exp(-((x + d)**2) / (2 * s**2))
                h_candidate = h_raw / np.max(h_raw)
                
                r = _ratio(h_candidate)
                
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_candidate.copy()

    # If no valid shape found (should not happen), return a single spike
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    else:
        # Ensure non-negative
        best_h = np.maximum(best_h, 0.0)
        
    return list(best_h)
