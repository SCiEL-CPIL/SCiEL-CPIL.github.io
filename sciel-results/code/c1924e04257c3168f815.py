import numpy as np
import time

N = 600

def compute_score(heights):
    """
    Computes the normalized maximum of the autoconvolution.
    Metric: max(h*h) / (||h||_1 * ||h||_inf)
    """
    h = np.array(heights, dtype=float)
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return float('inf')
        
    area = np.sum(h)
    max_val = np.max(h)
    
    if area == 0 or max_val == 0:
        return float('inf')
        
    # Discrete convolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    
    # Normalized metric
    # The constant function h=1 gives 1.0
    return max_conv / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_h = [1.0] * N
    best_score = compute_score(best_h)
    
    idx = np.arange(N)
    
    # Candidate 1: Constant (Seed)
    h_const = np.ones(N)
    score_const = compute_score(h_const)
    if score_const < best_score:
        best_score = score_const
        best_h = h_const
        
    # Candidate 2: Triangle (Peak in center)
    h_tri = 1.0 - np.abs(2.0 * idx / N - 1.0)
    h_tri = np.maximum(h_tri, 0)
    if np.max(h_tri) > 0:
        h_tri = h_tri / np.max(h_tri)
    score_tri = compute_score(h_tri)
    
    if score_tri < best_score:
        best_score = score_tri
        best_h = h_tri
        
    # Candidate 3: Sine Squared
    h_sin = np.sin(np.pi * idx / N) ** 2
    if np.max(h_sin) > 0:
        h_sin = h_sin / np.max(h_sin)
    score_sin = compute_score(h_sin)
    
    if score_sin < best_score:
        best_score = score_sin
        best_h = h_sin
        
    # Candidate 4: Gaussian
    center = N / 2.0
    sigma = N / 4.0
    x = (idx - center) / sigma
    h_gauss = np.exp(-0.5 * x**2)
    if np.max(h_gauss) > 0:
        h_gauss = h_gauss / np.max(h_gauss)
    score_gauss = compute_score(h_gauss)
    
    if score_gauss < best_score:
        best_score = score_gauss
        best_h = h_gauss

    # Candidate 5: Two Spikes at Ends
    h_two_spikes = np.zeros(N)
    h_two_spikes[:N//4] = 1.0
    h_two_spikes[3*N//4:] = 1.0
    score_two_spikes = compute_score(h_two_spikes)
    
    if score_two_spikes < best_score:
        best_score = score_two_spikes
        best_h = h_two_spikes

    # Simple local search: perturb the best candidate
    # Since the metric is scale invariant, we can normalize to max=1 after perturbation
    current_h = np.asarray(best_h, dtype=float)
    current_score = best_score
    
    # Try a few random perturbations
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline:
            break
        
        # Add small noise
        noise = np.random.normal(0, 0.1, N)
        h_new = current_h + noise
        h_new = np.maximum(h_new, 0)
        
        # Normalize to max 1
        if np.max(h_new) > 0:
            h_new = h_new / np.max(h_new)
            
        score_new = compute_score(h_new)
        if score_new < current_score:
            current_score = score_new
            current_h = h_new
        else:
            # Decay step size or accept with prob? Just keep best for now
            pass
            
    if current_score < best_score:
        best_score = current_score
        best_h = current_h

    return list(np.asarray(best_h).tolist())

# EVOLVE-BLOCK-END
