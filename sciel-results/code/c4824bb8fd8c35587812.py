import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    """
    Optimizes the step function heights to minimize the normalized maximum
    of the autoconvolution. We hypothesize the metric is scale-invariant and
    likely related to the L-infinity norm of the function, e.g.,
    J(h) = max(autoconv(h)) / (max(h))^2.
    This approach uses numerical optimization to find a local minimum.
    """
    
    def cost(h):
        # Ensure non-negative
        h = np.maximum(h, 0)
        if np.max(h) == 0:
            return 1e10
        # Compute discrete autoconvolution
        # c[k] = sum_i h[i] h[i+k]
        # Efficiently using FFT
        n_fft = 2 * N
        H = np.fft.rfft(h, n_fft)
        C = np.fft.irfft(H * H, n_fft)
        # The autoconvolution of a sequence of length N has length 2N-1
        # The peak is usually at the center (lag 0)
        max_c = np.max(C)
        max_h = np.max(h)
        return max_c / (max_h * max_h)

    # Initial guess: Triangular distribution (better than uniform for L_inf metric)
    h0 = np.linspace(1.0, 0.0, N // 2)
    h0 = np.concatenate([h0, h0[::-1]])[:N]
    
    # Bounds: h[i] >= 0
    bounds = [(0.0, 10.0)] * N
    
    # Optimize
    # Use L-BFGS-B for box constraints
    res = minimize(cost, h0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
    
    h_opt = res.x
    # Normalize to ensure max height is 1 (scale invariance allows this, but helps stability)
    if np.max(h_opt) > 0:
        h_opt = h_opt / np.max(h_opt)
        
    return h_opt.tolist()
