"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute full linear autoconvolution of x with itself."""
    n = len(x)
    # Using FFT for efficiency
    size = 2 * n
    X = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(X * X, n=size)[:n]
    return ac

def _score_candidate(x):
    """Compute max|autoconv| / sum(x^2). Returns inf if sum(x^2) is 0."""
    ss = np.sum(x * x)
    if ss < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    return np.max(np.abs(ac)) / ss

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_x = np.ones(N)
    best_score = _score_candidate(best_x)
    
    # Try several structured candidates
    candidates = [
        np.ones(N),
        np.random.default_rng(42).standard_normal(N),
        np.sign(np.random.default_rng(43).standard_normal(N)),
        np.ones(N) * np.array([1.0] * (N//2) + [-1.0] * (N - N//2)),
    ]
    
    # Add some random candidates
    rng = np.random.default_rng(123)
    for i in range(50):
        candidates.append(rng.standard_normal(N))
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    for x in candidates:
        if time.time() > deadline:
            break
        s = _score_candidate(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    # Try local optimization from best candidate using gradient-free methods
    if time.time() < deadline - 10:
        # Simple coordinate descent / random perturbations
        x = best_x.copy()
        cur_score = best_score
        rng2 = np.random.default_rng(999)
        
        for step in range(2000):
            if time.time() > deadline - 2:
                break
            
            # Random perturbation
            idx = rng2.integers(0, N)
            delta = rng2.standard_normal() * 0.1
            x_new = x.copy()
            x_new[idx] += delta
            
            s_new = _score_candidate(x_new)
            if s_new < cur_score:
                x = x_new
                cur_score = s_new
                best_x = x.copy()
                best_score = cur_score
    
    return best_x
# EVOLVE-BLOCK-END
