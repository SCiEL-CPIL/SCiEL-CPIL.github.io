"""Second autocorrelation inequality - maximize squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf) for nonnegative step heights."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution (linear convolution of h with reversed h, i.e., autocorrelation)
    # For step functions on equal intervals, the autoconvolution at each point
    # is the sum of products of overlapping steps.
    # Using FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * np.conj(H), n=n_fft)
    sup_ac = np.max(np.abs(ac))
    if sup_ac < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_ac)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Try various analytical patterns
    patterns = []
    
    # Uniform (baseline)
    patterns.append(np.ones(N))
    
    # Triangular
    x = np.linspace(0, 1, N)
    patterns.append(np.abs(1 - 2 * x))
    
    # Cosine bell
    patterns.append(np.cos(np.pi * (np.arange(N) / (N - 1) - 0.5)) ** 2)
    
    # Gaussian-like
    center = (N - 1) / 2.0
    sigma = N / 4.0
    patterns.append(np.exp(-0.5 * ((np.arange(N) - center) / sigma) ** 2))
    
    # Sparse: only middle portion
    for frac in [0.3, 0.5, 0.7]:
        k = max(1, int(N * frac))
        h = np.zeros(N)
        h[(N - k) // 2 : (N + k) // 2] = 1.0
        patterns.append(h)
    
    # Alternating blocks
    for block_size in [2, 3, 5, 7, 10]:
        h = np.zeros(N)
        for i in range(0, N, block_size * 2):
            h[i:i + block_size] = 1.0
        if np.sum(h) > 0:
            patterns.append(h)
    
    # Evaluate all patterns
    for p in patterns:
        if np.any(p < 0):
            p = np.clip(p, 0, None)
        if np.sum(p) < 1e-15:
            continue
        s = _compute_ratio(p)
        if s > best_score:
            best_score = s
            best_h = p.copy()
    
    # Local optimization: perturb and keep improvements
    rng = np.random.default_rng(42)
    h = best_h.copy()
    current_score = best_score
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Random perturbation
        delta = rng.normal(0, 0.1, size=N)
        h_new = np.clip(h + delta, 0, None)
        
        if np.sum(h_new) < 1e-15:
            continue
        
        new_score = _compute_ratio(h_new)
        if new_score > current_score:
            h = h_new
            current_score = new_score
            best_h = h.copy()
            best_score = current_score
    
    # Final local refinement with smaller steps
    for step_size in [0.05, 0.01, 0.005]:
        for iteration in range(2000):
            if time.time() > deadline:
                break
            idx = rng.integers(0, N)
            h_new = h.copy()
            h_new[idx] = max(0, h[idx] + rng.normal(0, step_size))
            new_score = _compute_ratio(h_new)
            if new_score > current_score:
                h = h_new
                current_score = new_score
                best_h = h.copy()
                best_score = current_score
    
    # Ensure nonnegative and return as list of floats
    result = np.clip(best_h, 0, None).tolist()
    return result
