"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute full linear autoconvolution of x with itself."""
    n = len(x)
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

    # Start with random ±1 candidates - these have score ~0.13 vs seed 1.0
    rng = np.random.default_rng(42)
    
    # Generate many random ±1 candidates and pick the best
    best_x = None
    best_score = float('inf')
    
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        s = _score_candidate(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    # Local optimization: coordinate descent by flipping signs
    # Flipping sign at index k: x[k] -> -x[k], change = -2*x[k] at position k
    # This changes autoconv by -2*x[k] * autoconv_of_delta_k
    # For efficiency, just recompute
    
    x = best_x.copy()
    cur_score = best_score
    rng2 = np.random.default_rng(999)
    
    # Precompute: for each index k, what's the effect of flipping?
    # autoconv of delta_k (1 at position k, 0 elsewhere) convolved with x
    # = x shifted by k (for k < N) and reversed (for k >= N)
    # Actually, let's just do direct coordinate descent with recomputation
    
    # Phase 1: Random coordinate descent
    step = 0
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        step += 1
        idx = rng2.integers(0, N)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]  # flip sign
        
        s_new = _score_candidate(x_new)
        if s_new < cur_score:
            x = x_new
            cur_score = s_new
            best_x = x.copy()
            best_score = cur_score
            no_improve = 0
        else:
            no_improve += 1
        
        # Occasionally try small perturbations on continuous values
        if step % 50 == 0 and time.time() < deadline - 10:
            idx2 = rng2.integers(0, N)
            x_temp = x.copy()
            x_temp[idx2] += rng2.standard_normal() * 0.5
            s_temp = _score_candidate(x_temp)
            if s_temp < cur_score:
                x = x_temp
                cur_score = s_temp
                best_x = x.copy()
                best_score = cur_score
                no_improve = 0
    
    # Phase 2: More systematic coordinate descent - sweep through all indices
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for idx in range(N):
            if time.time() > deadline - 3:
                break
            x_new = x.copy()
            x_new[idx] = -x_new[idx]
            s_new = _score_candidate(x_new)
            if s_new < cur_score:
                x = x_new
                cur_score = s_new
                best_x = x.copy()
                best_score = cur_score
                improved = True
    
    return best_x
# EVOLVE-BLOCK-END
