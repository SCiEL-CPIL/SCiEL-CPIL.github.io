"""Optimized program for signed autocorrelation minimum peak-to-sum-ratio."""
import numpy as np
import time

N = 400

def compute_objective(x):
    """Compute max(|autoconv|) / (sum(x))^2"""
    # Autoconvolution via FFT
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    autoconv = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
    peak = np.max(np.abs(autoconv))
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return peak / (s * s)

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_x = None
    best_val = 1e10
    
    # Try several initializations and local search
    rng = np.random.default_rng(42)
    
    # Initial candidates
    candidates = []
    
    # All ones
    candidates.append(np.ones(N))
    
    # Random ±1 sequences
    for _ in range(20):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Structured: first half +1, second half -1 with smooth transition
    t = np.linspace(-1, 1, N)
    candidates.append(np.sign(t))
    
    # Gradient-like search on continuous values
    for x0 in candidates:
        if time.time() > deadline:
            break
            
        x = x0.copy()
        # Normalize to have reasonable sum
        s = np.sum(x)
        if abs(s) < 1:
            x = x * 10 / max(abs(s), 1e-6)
        
        val = compute_objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Local search: try flipping small perturbations
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random index to perturb
            idx = rng.integers(0, N)
            old_val = x[idx]
            x[idx] = -old_val  # Flip sign
            
            new_val = compute_objective(x)
            if new_val < val:
                val = new_val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
            else:
                x[idx] = old_val
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
