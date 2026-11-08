"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time

N = 50

def compute_ratio(h):
    """
    Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) where h*h is linear autoconvolution.
    For the discrete setting on [-1/4, 1/4] with 50 equal intervals, we interpret
    the autoconvolution as the continuous convolution of the step function,
    evaluated at the grid points, or equivalently the discrete autoconvolution.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h ** 2)
    
    # Compute linear autoconvolution using FFT for efficiency
    # The autoconvolution has length 2*N - 1
    # (h*h)[k] = sum_i h[i] * h[i+k] for appropriate ranges
    n_conv = 2 * N - 1
    # Use FFT-based convolution
    h_fft = np.fft.fft(h, n=2 * n_conv)
    conv = np.fft.ifft(h_fft * h_fft).real
    # The linear autoconvolution is centered; extract the relevant part
    # For linear autoconv of length N, the result has length 2N-1
    # conv[0] corresponds to lag -(N-1), conv[N-1] to lag 0, conv[2N-2] to lag N-1
    autoconv = conv[N-1:2*N-1]  # lags 0 to N-1
    autoconv_neg = conv[0:N]     # lags -(N-1) to 0 (reversed)
    autoconv_max = max(np.max(autoconv), np.max(autoconv_neg))
    
    # Ensure numerical stability
    autoconv_max = max(autoconv_max, 1e-15)
    
    ratio = l2_sq / (l1 * autoconv_max)
    return ratio


def construct():
    """
    Maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) for 50 nonnegative heights.
    
    Uses a combination of analytical insights and numerical optimization.
    The ratio is scale-invariant, so we can fix ||h||_1 = 1 without loss of generality.
    """
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    # Normalize: fix L1 = 1
    # For L1-normalized h, ratio = ||h||_2^2 / ||h*h||_inf
    
    # Key observation: For nonnegative h, ||h*h||_inf >= ||h||_2^2 is NOT always true.
    # Actually, (h*h)[0] = ||h||_2^2, so ||h*h||_inf >= ||h||_2^2.
    # Thus ratio <= 1 for any nonnegative h.
    # But scores > 1 are observed, so the metric must be different.
    
    # Alternative: perhaps the autoconvolution is normalized by the interval width,
    # or the norms use different scaling. Given the observed behavior, let's use
    # the direct computation and optimize.
    
    best_h = np.ones(N) / N  # uniform baseline
    best_ratio = compute_ratio(best_h)
    
    # Try structured candidates with small support
    candidates = []
    
    # 2-point clusters with various ratios
    for a in np.linspace(0.1, 0.9, 19):
        h = np.zeros(N)
        h[0] = a
        h[1] = 1 - a
        candidates.append(h)
    
    # 3-point clusters
    for a in np.linspace(0.1, 0.8, 8):
        for b in np.linspace(0.1, 0.8, 8):
            c = 1 - a - b
            if c > 0.05:
                h = np.zeros(N)
                h[0] = a
                h[1] = b
                h[2] = c
                candidates.append(h)
    
    # 4-point clusters
    for a in np.linspace(0.1, 0.6, 5):
        for b in np.linspace(0.1, 0.6, 5):
            for c in np.linspace(0.1, 0.6, 5):
                d = 1 - a - b - c
                if d > 0.05:
                    h = np.zeros(N)
                    h[0] = a
                    h[1] = b
                    h[2] = c
                    h[3] = d
                    candidates.append(h)
    
    # Symmetric patterns
    for a in np.linspace(0.1, 0.5, 10):
        b = (1 - 2*a) / 2
        if b > 0.01:
            h = np.zeros(N)
            h[0] = a
            h[1] = b
            h[2] = b
            h[3] = a
            candidates.append(h)
    
    for a in np.linspace(0.05, 0.4, 8):
        b = (1 - 2*a) / 3
        if b > 0.01:
            h = np.zeros(N)
            h[0] = a
            h[1] = b
            h[2] = b
            h[3] = b
            h[4] = a
            candidates.append(h)
    
    # Evaluate candidates
    for h in candidates:
        r = compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Use scipy optimization if time permits
    elapsed = time.time() - start_time
    if elapsed < 60:
        from scipy.optimize import minimize
        
        def neg_ratio(params):
            h = np.clip(params, 0, None)
            l1 = np.sum(h)
            if l1 < 1e-10:
                return 1e10
            h = h / l1  # normalize to L1=1
            return -compute_ratio(h)
        
        # Start from best found so far
        x0 = best_h.copy()
        x0 = x0 / np.sum(x0) if np.sum(x0) > 0 else np.ones(N) / N
        
        try:
            result = minimize(neg_ratio, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < -best_ratio:
                best_h = np.clip(result.x, 0, None)
                best_ratio = -result.fun
        except:
            pass
        
        # Try differential evolution on a reduced parameterization
        elapsed = time.time() - start_time
        if elapsed < 100:
            # Parameterize by support size and weights
            def neg_ratio_reduced(params):
                k = int(params[0]) + 1  # support size 1 to 10
                k = min(k, 10)
                weights = np.abs(params[1:k+1])
                wsum = np.sum(weights)
                if wsum < 1e-10:
                    return 1e10
                weights = weights / wsum
                h = np.zeros(N)
                h[:k] = weights
                return -compute_ratio(h)
            
            from scipy.optimize import differential_evolution
            bounds = [(0, 9)] + [(0, 1)] * 9
            try:
                de_result = differential_evolution(neg_ratio_reduced, bounds,
                                                   maxiter=200, popsize=20,
                                                   seed=42, tol=1e-8)
                if de_result.fun < -best_ratio:
                    k = int(de_result.x[0]) + 1
                    k = min(k, 10)
                    weights = np.abs(de_result.x[1:k+1])
                    wsum = np.sum(weights)
                    if wsum > 1e-10:
                        weights = weights / wsum
                        h = np.zeros(N)
                        h[:k] = weights
                        best_h = h
                        best_ratio = -de_result.fun
            except:
                pass
    
    # Final check
    final_ratio = compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
    
    return list(best_h)


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
