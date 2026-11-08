"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time

N = 50

def compute_ratio(h):
    """
    Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) where h*h is linear autoconvolution.
    For the discrete setting on [-1/4, 1/4] with 50 equal intervals, we interpret
    the autoconvolution as the discrete autoconvolution.
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
    n_conv = 2 * N - 1
    h_fft = np.fft.fft(h, n=2 * n_conv)
    conv = np.fft.ifft(h_fft * h_fft).real
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
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # The ratio is scale-invariant, so we can fix ||h||_1 = 1.
    # For L1-normalized h, ratio = ||h||_2^2 / ||h*h||_inf.
    # Note: (h*h)[0] = ||h||_2^2, so ratio <= 1 for standard autoconvolution.
    # However, the task implies scores > 1 are possible, suggesting a different
    # normalization or interpretation. We optimize the ratio as defined.
    
    best_h = np.ones(N) / N
    best_ratio = compute_ratio(best_h)
    
    # Generate diverse candidates
    candidates = []
    
    # 1. Single spike
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        candidates.append(h)
    
    # 2. Two spikes with various separations and weights
    for i in range(N):
        for j in range(i+1, N):
            for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                h = np.zeros(N)
                h[i] = w
                h[j] = 1.0 - w
                candidates.append(h)
    
    # 3. Gaussian-like profiles
    centers = np.linspace(5, N-5, 10)
    sigmas = np.linspace(1, 10, 10)
    for c in centers:
        for s in sigmas:
            h = np.exp(-0.5 * ((np.arange(N) - c) / s) ** 2)
            h = h / np.sum(h)
            candidates.append(h)
    
    # 4. Uniform blocks
    for start in range(0, N-1, 5):
        for length in range(2, N-start, 5):
            h = np.zeros(N)
            h[start:start+length] = 1.0 / length
            candidates.append(h)
    
    # 5. Random sparse vectors
    np.random.seed(42)
    for _ in range(200):
        k = np.random.randint(1, 10)
        indices = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0 / k
        candidates.append(h)
    
    # 6. Random dense vectors
    for _ in range(200):
        h = np.random.rand(N)
        h = h / np.sum(h)
        candidates.append(h)
    
    # Evaluate candidates
    for h in candidates:
        r = compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Use scipy optimization if time permits
    elapsed = time.time() - start_time
    if elapsed < 120:
        from scipy.optimize import minimize, differential_evolution
        
        def neg_ratio(params):
            h = np.clip(params, 0, None)
            l1 = np.sum(h)
            if l1 < 1e-10:
                return 1e10
            h = h / l1
            return -compute_ratio(h)
        
        # Start from best found so far
        x0 = best_h.copy()
        
        try:
            result = minimize(neg_ratio, x0, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < -best_ratio:
                best_h = np.clip(result.x, 0, None)
                best_ratio = -result.fun
        except:
            pass
        
        # Try differential evolution on a reduced parameterization
        elapsed = time.time() - start_time
        if elapsed < 130:
            def neg_ratio_reduced(params):
                k = int(params[0]) + 1
                k = min(k, 15)
                weights = np.abs(params[1:k+1])
                wsum = np.sum(weights)
                if wsum < 1e-10:
                    return 1e10
                weights = weights / wsum
                h = np.zeros(N)
                h[:k] = weights
                return -compute_ratio(h)
            
            bounds = [(0, 14)] + [(0, 1)] * 14
            try:
                de_result = differential_evolution(neg_ratio_reduced, bounds,
                                                   maxiter=100, popsize=30,
                                                   seed=42, tol=1e-8)
                if de_result.fun < -best_ratio:
                    k = int(de_result.x[0]) + 1
                    k = min(k, 15)
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
