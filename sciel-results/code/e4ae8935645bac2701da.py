"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _value(h):
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def construct():
    start = time.time()
    deadline = 120.0
    
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Try various smooth positive shapes
    i = np.arange(N)
    
    # Gaussian-like
    for sigma_frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if time.time() - start > deadline - 5:
            break
        center = N / 2
        h = np.exp(-0.5 * ((i - center) / (sigma_frac * N / 2)) ** 2)
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Raised cosine
    for width_frac in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if time.time() - start > deadline - 5:
            break
        h = 0.5 * (1 + np.cos(2 * np.pi * i / (N / width_frac)))
        h[i > N / width_frac] = 0
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Triangular
    for width_frac in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if time.time() - start > deadline - 5:
            break
        w = int(N * width_frac)
        h = np.zeros(N)
        if w >= 2:
            h[:w] = np.minimum(np.arange(w) / (w / 2), (w - np.arange(w)) / (w / 2))
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Simple optimization: try to reduce peak by flattening
    # Use a power law shape: h[i] = (1 - |2i/N - 1|)^p
    for p in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        if time.time() - start > deadline - 5:
            break
        h = np.power(np.maximum(0, 1 - np.abs(2 * i / N - 1)), p)
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
