import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START

def calculate_objective(heights):
    """
    Calculates the normalized maximum of the autoconvolution.
    Assumes:
    - f is a step function on N intervals of equal width delta.
    - Domain is [-1/4, 1/4].
    - Normalization is ||f*f||_inf / ||f||_2^2.
    """
    delta = (0.5) / N  # Total width 1/4 - (-1/4) = 1/2
    
    # L2 norm squared
    norm2_sq = np.sum(heights**2) * delta
    
    if norm2_sq < 1e-12:
        return float('inf')
        
    # Autoconvolution
    conv = np.convolve(heights, heights, mode='full')
    
    max_conv = np.max(conv) * (delta**2)
    
    return max_conv / norm2_sq

def construct():
    start_time = time.time()
    timeout = 115.0  # Keep 5s buffer
    
    # Generate candidates
    candidates = []
    
    # 1. Uniform (likely optimal or near-optimal)
    candidates.append(np.ones(N))
    
    # 2. Triangular (centered)
    x = np.linspace(-0.5, 0.5, N)
    candidates.append(np.abs(1 - 2*np.abs(x)))
    
    # 3. Gaussian
    candidates.append(np.exp(-x**2 * 10))
    candidates.append(np.exp(-x**2 * 50))
    
    # 4. Edge peaks
    edge = np.ones(N) * 1e-6
    edge[:N//4] = 1.0
    edge[3*N//4:] = 1.0
    candidates.append(edge)
    
    best_heights = None
    best_obj = float('inf')
    
    for c in candidates:
        if time.time() - start_time > timeout:
            break
        c = c.copy()
        obj = calculate_objective(c)
        if obj < best_obj:
            best_obj = obj
            best_heights = c
            
    if best_heights is None:
        best_heights = np.ones(N)
        
    return list(best_heights)

# EVOLVE-BLOCK-END
