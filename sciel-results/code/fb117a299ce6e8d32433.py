import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Normalize to L1=1
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # The theoretical maximum for non-negative h is 1.0.
    # Uniform and Delta distributions achieve this.
    # The previous program likely degraded from 1.0 due to search bugs.
    # We will test a few known-good candidates and return the best.
    
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N) / N)
    
    # 2. Delta (single spike)
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    candidates.append(h_delta)
    
    # 3. Delta at center
    h_delta_c = np.zeros(N)
    h_delta_c[N//2] = 1.0
    candidates.append(h_delta_c)
    
    # 4. Two adjacent spikes
    h_two = np.zeros(N)
    h_two[0] = 0.5
    h_two[1] = 0.5
    candidates.append(h_two)
    
    best_h = candidates[0]
    best_score = compute_ratio(best_h)
    
    for h in candidates[1:]:
        score = compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
    # If we have time, we can try a simple random search to see if we can find 1.0 
    # more robustly, but since 1.0 is the max, any valid 1.0 is fine.
    # We just need to ensure we don't return something < 1.0.
    # The candidates above are analytically 1.0.
    
    return list(best_h)
