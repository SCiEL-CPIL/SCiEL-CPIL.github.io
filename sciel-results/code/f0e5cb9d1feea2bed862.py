import numpy as np
import time

N = 50
DX = 0.5 / N  # interval width = (1/4 - (-1/4)) / 50 = 0.5/50

def compute_ratio(h):
    """Compute the ratio (L2^2) / (L1 * sup_autoconv)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.dot(h, h)  # sum h_i^2
    l1 = np.sum(h)
    if l1 == 0 or l2_sq == 0:
        return 0.0
    # Autoconvolution: c_k = sum_i h_i * h_{i+k}
    # Use FFT for efficiency
    n = len(h)
    # Zero-pad to avoid circular convolution
    size = 2 * n
    h_fft = np.fft.rfft(h, size)
    autoconv = np.fft.irfft(h_fft * np.conj(h_fft), size)[:n]
    sup_autoconv = np.max(np.abs(autoconv))
    if sup_autoconv == 0:
        return 0.0
    return l2_sq / (l1 * sup_autoconv)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Try a few heuristic initializations
    candidates = [
        np.ones(N),
        np.array([1.0] + [0.0]*(N-1)),
        np.array([1.0, 1.0] + [0.0]*(N-2)),
        np.linspace(1.0, 0.0, N),
        np.linspace(0.0, 1.0, N),
        np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)]),
    ]
    
    for cand in candidates:
        score = compute_ratio(cand)
        if score > best_score:
            best_score = score
            best_h = cand.copy()
    
    # Simple local search: perturb and keep improvements
    rng = np.random.default_rng(42)
    current_h = best_h.copy()
    current_score = best_score
    
    step_size = 0.1
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Random perturbation
        perturbation = rng.normal(0, step_size, N)
        new_h = np.maximum(0, current_h + perturbation)
        
        new_score = compute_ratio(new_h)
        if new_score > current_score:
            current_h = new_h
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_h = current_h.copy()
            step_size = max(0.01, step_size * 0.95)
        else:
            step_size = max(0.01, step_size * 0.98)
    
    return list(best_h)
