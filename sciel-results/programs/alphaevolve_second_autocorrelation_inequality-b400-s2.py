"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import numpy as np
from scipy.optimize import minimize
import time

def _score(h):
    """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step heights h."""
    h = np.array(h, dtype=float)
    n = len(h)
    w = 1.0 / (2.0 * n)  # interval width: [-1/4, 1/4] divided into n intervals
    
    # L2^2 and L1
    l2_sq = np.sum(h**2) * w
    l1 = np.sum(h) * w
    
    if l1 < 1e-15:
        return 0.0
    
    # Autoconvolution: (f*f)(t) = sum_i sum_j h_i h_j * overlap(t, interval_i, interval_j)
    # For step functions, we can compute this efficiently
    # The autoconvolution is a piecewise linear function
    # We need its maximum value
    
    # Compute autoconvolution on a fine grid
    # Support is [-1/2, 1/2]
    n_grid = 10001
    t = np.linspace(-0.5, 0.5, n_grid)
    dt = t[1] - t[0]
    
    # f(t) values on grid
    # Intervals are centered at: -1/4 + (i + 0.5)*w for i=0..n-1
    centers = -0.25 + (np.arange(n) + 0.5) * w
    f_grid = np.zeros(n_grid)
    for i in range(n):
        lo = centers[i] - w/2
        hi = centers[i] + w/2
        mask = (t >= lo) & (t < hi)
        f_grid[mask] += h[i]
    
    # Convolve using FFT
    f_fft = np.fft.fft(f_grid, n=2*n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    max_conv = np.max(conv)
    
    if max_conv < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with single spike (known to give ratio ~100)
    best_h = np.zeros(N)
    best_h[25] = 1.0  # Middle spike
    best_score = _score(best_h)
    
    # Try various profiles
    candidates = []
    
    # Single spike at different positions
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            candidates.append(h)
    
    # Triangular profiles
    for width in [1, 2, 3, 5, 10]:
        for center in range(width//2, N - width//2):
            h = np.zeros(N)
            for i in range(N):
                dist = abs(i - center)
                if dist <= width//2:
                    h[i] = 1.0 - 2.0*dist/width
            if np.sum(h) > 0:
                candidates.append(h)
    
    # Gaussian-like profiles
    for sigma in [1, 2, 3, 5]:
        for center in [N//4, N//2, 3*N//4]:
            i = np.arange(N)
            h = np.exp(-0.5 * ((i - center)/sigma)**2)
            candidates.append(h)
    
    # Evaluate candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization from best candidate
    if time.time() < deadline - 5:
        def neg_score(h):
            h = np.abs(h)  # Enforce nonnegativity
            return -_score(h)
        
        result = minimize(neg_score, best_h, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        
        if result.success or -result.fun > best_score:
            h_opt = np.abs(result.x)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
    
    return list(best_h)
# EVOLVE-BLOCK-END
