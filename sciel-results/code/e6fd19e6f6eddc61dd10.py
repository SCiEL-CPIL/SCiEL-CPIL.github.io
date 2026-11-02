import numpy as np
import time
from scipy.optimize import minimize

N = 600

def _compute_max_autoconv(h):
    """
    Compute the maximum autoconvolution value excluding lag 0.
    h is a vector of length N.
    Returns the max value of the autoconvolution at non-zero lags.
    """
    if np.sum(h) < 1e-12:
        return 0.0
    # Efficiently compute autoconvolution using FFT
    fft_size = 1 << (2 * N - 1).bit_length()
    h_fft = np.fft.fft(h, n=fft_size)
    conv_fft = h_fft * h_fft
    conv = np.fft.ifft(conv_fft).real
    
    # Ensure numerical stability
    conv = np.maximum(conv, 0)
    
    # Exclude center (lag 0)
    if N == 1:
        return 0.0
        
    max_val = np.max(np.concatenate([conv[:N-1], conv[N:]]))
    return max_val

def _compute_score(h):
    """
    Normalized score: max_autoconv / (sum h)^2
    """
    s = np.sum(h)
    if s < 1e-12:
        return np.inf
    return _compute_max_autoconv(h) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Strategy: "Zero-SideloBe" Design via Sparse Support.
    # The previous approach used dense, smooth functions (Gaussian, etc.).
    # A structurally different approach is to use a "Sidelobe Cancellation" design,
    # similar to Barker codes or Golay sequences, but adapted for real non-negative functions.
    # We aim to construct a function h that has significant mass only at specific locations
    # such that the cross-correlations (autoconvolution at non-zero lags) cancel out or are minimized.
    
    # Key Insight:
    # If we can find a set of indices S such that the autocorrelation R(k) = sum_{i in S} h_i h_{i+k} is 0 for all k != 0,
    # then the score is 0. This is impossible for real non-negative h with support > 1 (R(1) >= 0).
    # However, we can minimize the *peak* sidelobe by spreading the energy in a way that the sum of products is small.
    
    # Alternative Structural Idea: "Complementary" Pairs.
    # Golay sequences have complementary autocorrelation properties.
    # For a step function, we can try to design h to look like a "sum of shifted rectangles" that destructively interfere in the correlation domain.
    
    # Let's try a "Binary-like" or "Sparse" approach.
    # Suppose we divide the domain into M blocks. We assign heights to these blocks.
    # If we make the heights alternate or follow a specific pattern, we might reduce sidelobes.
    
    # A very powerful family is the "Prolate Spheroidal Wave Function" (PSWF) or similar time-frequency limited functions.
    # But let's try a simpler "Multi-Block" approach with optimized amplitudes.
    
    # Let's try a design based on the "Minimum Peak Sidelobe" (MPSL) sequence concept.
    # For real non-negative sequences, the optimal is often close to a "raised cosine" or "Hann" window,
    # but we can do better by using a "Superimposed" structure.
    
    # Let's try a "Fourier Series" representation with a few terms, optimized for low sidelobes.
    # h(t) = a0 + a1*cos(2*pi*t) + a2*cos(4*pi*t) + ...
    # We constrain h(t) >= 0.
    
    # Let's use a parametric family with 3-4 terms.
    # t in [-0.5, 0.5] (scaled to [-1/4, 1/4] later)
    
    centers = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = (centers + 0.25) # Map to [0, 0.5]
    t = t / 0.5 # Map to [0, 1]
    t = t - 0.5 # Map to [-0.5, 0.5]
    
    # Family: h(t) = max(0, 1 - a * (2*t)^2 + b * (2*t)^4)
    # This is a "Polynomial Window" of order 4.
    # We can optimize a and b.
    
    def _eval_poly(params):
        a, b = params
        h = 1 - a * (2*t)**2 + b * (2*t)**4
        h = np.maximum(h, 0)
        if np.sum(h) < 1e-9:
            return 1e9
        return _compute_score(h)
    
    best_score = _compute_score(np.ones(N) / N)
    best_h = np.ones(N) / N
    
    # Optimize the polynomial window
    # Initial guess: a=1, b=0 (Parabola)
    try:
        result = minimize(_eval_poly, [1.0, 0.0], method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
        if result.fun < best_score:
            best_score = result.fun
            a, b = result.x
            best_h = np.maximum(0, 1 - a * (2*t)**2 + b * (2*t)**4)
    except Exception as e:
        pass
    
    # Try a different family: "Sum of Two Gaussian Bumps"
    # h(t) = exp(-alpha * (t - mu1)^2) + exp(-alpha * (t - mu2)^2)
    # This allows for a "dip" in the center, which can reduce the peak sidelobe at small lags.
    
    def _eval_two_gauss(params):
        alpha, mu1, mu2 = params
        if alpha < 0:
            return 1e9
        h = np.exp(-alpha * (t - mu1)**2) + np.exp(-alpha * (t - mu2)**2)
        if np.sum(h) < 1e-9:
            return 1e9
        return _compute_score(h)
    
    # Initial guess: symmetric bumps
    try:
        result = minimize(_eval_two_gauss, [5.0, -0.2, 0.2], method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
        if result.fun < best_score:
            best_score = result.fun
            alpha, mu1, mu2 = result.x
            best_h = np.exp(-alpha * (t - mu1)**2) + np.exp(-alpha * (t - mu2)**2)
    except Exception as e:
        pass
    
    # Try a "Cosine Window" with optimized bandwidth
    # h(t) = 0.5 * (1 + cos(2*pi*t / W)) for |t| < W/2, 0 otherwise
    # This is a "Raised Cosine" with adjustable width W.
    
    def _eval_raised_cos(params):
        W = params[0]
        if W <= 0 or W > 1:
            return 1e9
        h = np.zeros(N)
        mask = np.abs(t) < W/2
        h[mask] = 0.5 * (1 + np.cos(2*np.pi*t[mask] / W))
        if np.sum(h) < 1e-9:
            return 1e9
        return _compute_score(h)
    
    try:
        result = minimize(_eval_raised_cos, [1.0], method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        if result.fun < best_score:
            best_score = result.fun
            W = result.x[0]
            h = np.zeros(N)
            mask = np.abs(t) < W/2
            best_h = 0.5 * (1 + np.cos(2*np.pi*t[mask] / W)) * mask
            # Need to handle the mask correctly in the assignment
            h_full = np.zeros(N)
            h_full[mask] = 0.5 * (1 + np.cos(2*np.pi*t[mask] / W))
            best_h = h_full
    except Exception as e:
        pass
    
    # If time permits, try a "Random Search" with local refinement.
    # Generate random non-negative vectors, normalize, and check score.
    # This is unlikely to beat the optimized parametric families, but it's a different strategy.
    
    if time.time() < deadline - 10:
        # Try a few random "block" structures
        for _ in range(10):
            # Random number of blocks
            n_blocks = np.random.randint(2, 10)
            # Random block widths
            widths = np.random.dirichlet(np.ones(n_blocks) * 2) * 0.5
            # Random positions
            positions = np.random.uniform(-0.25, 0.25, n_blocks)
            # Random amplitudes
            amplitudes = np.random.uniform(0.5, 1.5, n_blocks)
            
            h = np.zeros(N)
            for i in range(n_blocks):
                center = positions[i]
                w = widths[i]
                amp = amplitudes[i]
                mask = np.abs(centers - center) < w/2
                h[mask] += amp
            
            sc = _compute_score(h)
            if sc < best_score:
                best_score = sc
                best_h = h
    
    return list(best_h)
