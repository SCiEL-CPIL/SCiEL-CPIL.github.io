import numpy as np
import time
import scipy.optimize as opt

N = 600
# Precompute FFT size for linear convolution of length N with itself
# Result length is 2*N - 1. We need an FFT size >= 2*N - 1.
FFT_SIZE = 1 << (2*N - 1).bit_length()

# Precompute a fine grid for accurate evaluation of the peak
FINE_N = 20000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to interval indices
W = 0.5 / N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT_FINE = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Metric: max(g) / sum(h^2)
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # L2 norm squared
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return float('inf')
        
    # Expand h to fine grid
    f = h[CELL]
    
    # FFT convolution
    F = np.fft.rfft(f, N_FFT_FINE)
    conv = np.fft.irfft(F * F, N_FFT_FINE)[:2 * FINE_N - 1] * DX
    
    peak = np.max(conv)
    return float(peak / l2_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for 150s limit
    
    # Phase 1: Initialize with a few candidate shapes
    # Uniform
    h_uniform = np.ones(N)
    # Gaussian centered
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h_gauss = np.exp(-20 * x**2)
    # Cosine bump (positive on [-1/4, 1/4])
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * x))  # peaks at 0, zero at ends
    
    best_h = h_uniform.copy()
    best_val = ratio_from_h(h_uniform)
    
    for h_init in [h_gauss, h_cos]:
        v = ratio_from_h(h_init)
        if v < best_val:
            best_val = v
            best_h = h_init.copy()
    
    # Phase 2: Coordinate Descent / Local Search on the 600 dimensions
    # We will use a simple stochastic local search with adaptive step sizes.
    # Since N=600 is manageable, we can try optimizing subsets of coordinates or using gradient estimates.
    # However, the objective is non-smooth (max of linear functions).
    # A simple approach: random coordinate updates.
    
    rng = np.random.default_rng(42)
    current_h = best_h.copy()
    current_val = best_val
    
    # Precompute neighbors for faster evaluation? No, FFT is fast.
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Choose a random block of coordinates to perturb
        block_size = int(rng.integers(1, 10))
        start_idx = int(rng.integers(0, N - block_size + 1))
        end_idx = start_idx + block_size
        
        # Perturb the block
        h_try = current_h.copy()
        # Small random additive noise
        noise = rng.normal(0, 0.01, size=block_size)
        h_try[start_idx:end_idx] += noise
        h_try = np.clip(h_try, 0.0, None)
        
        v = ratio_from_h(h_try)
        if v < current_val:
            current_val = v
            current_h = h_try
            if current_val < best_val:
                best_val = current_val
                best_h = h_try.copy()
    
    # Phase 3: Simulated Annealing-like refinement
    # If we have time, do a more structured search.
    if time.time() < deadline - 10.0:
        temp = 0.01
        for it in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            h_try = current_h.copy()
            # Random block
            block_size = int(rng.integers(5, 20))
            start_idx = int(rng.integers(0, N - block_size + 1))
            end_idx = start_idx + block_size
            
            # Additive Gaussian bump
            center = (start_idx + end_idx) // 2
            width = block_size // 2
            xs = np.linspace(-1, 1, block_size)
            bump = np.exp(-xs**2) * temp
            h_try[start_idx:end_idx] += bump
            h_try = np.clip(h_try, 0.0, None)
            
            v = ratio_from_h(h_try)
            delta = v - current_val
            if delta < 0 or np.random.rand() < np.exp(-delta / (temp * 0.1)):
                current_val = v
                current_h = h_try
                if current_val < best_val:
                    best_val = current_val
                    best_h = h_try.copy()
            temp *= 0.999
    
    return list(best_h)
