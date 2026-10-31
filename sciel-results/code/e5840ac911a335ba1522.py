"""Optimized signed step function for C3' minimization."""
import numpy as np
from scipy import signal
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution via FFT."""
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f * np.conj(f), n=2*n)
    return ac[:n]  # return first n elements (k=0 to n-1)

def _score(x):
    """Compute max |autoconv| / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(x)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Strategy: start with a good base and optimize
    # Use a construction based on alternating blocks with refinement
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)
    
    # Try several initial patterns
    patterns = [
        np.ones(N),
        np.sign(np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))),
        np.sign(np.cos(np.linspace(0, 2*np.pi, N, endpoint=False))),
        np.random.RandomState(42).randint(-1, 2, N).astype(float),
    ]
    
    for p in patterns:
        s = np.sum(p)
        if abs(s) > 1e-6:
            sc = _score(p)
            if sc < best_score:
                best_score = sc
                best_x = p.copy()
    
    # Coordinate descent optimization
    x = best_x.copy()
    lr = 0.1
    for iteration in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        # Compute autoconvolution
        ac = _autoconv(x)
        s = np.sum(x)
        if abs(s) < 1e-12:
            break
        
        # Gradient of max|ac| w.r.t. x is tricky, use subgradient
        # Instead, do a simple perturbation-based local search
        improved = False
        for _ in range(100):
            idx = np.random.randint(0, N)
            old_val = x[idx]
            # Try flipping sign or adjusting
            x[idx] = -old_val
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
                improved = True
            else:
                x[idx] = old_val
        
        if not improved:
            # Try small continuous adjustments
            for _ in range(50):
                idx = np.random.randint(0, N)
                old_val = x[idx]
                x[idx] = old_val + np.random.randn() * 0.1
                sc = _score(x)
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
                else:
                    x[idx] = old_val
    
    # Final polish: try to snap to ±1 if it improves
    x_bin = np.sign(best_x)
    s_bin = np.sum(x_bin)
    if abs(s_bin) > 1e-6:
        sc_bin = _score(x_bin)
        if sc_bin < best_score:
            best_x = x_bin
            best_score = sc_bin
    
    return best_x
