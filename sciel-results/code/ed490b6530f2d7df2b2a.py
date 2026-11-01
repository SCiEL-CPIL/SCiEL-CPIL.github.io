"""Rewritten program for third autocorrelation inequality task."""
import numpy as np
import time

N = 400

def _objective(h):
    """Compute the upper bound on C3 for signed step function h.
    Normalized max absolute autocorrelation.
    """
    n = N
    size = 2 * n
    H = np.zeros(size, dtype=np.float64)
    H[:n] = h
    F = np.fft.fft(H, n=size)
    ac = np.fft.ifft(F * np.conj(F)).real[:n]
    # Normalize: for all-ones, ac[0] = n, so ac/n has max 1.0
    return np.max(np.abs(ac)) / n

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Verify seed
    seed_val = _objective(np.ones(N))
    # Should be ~1.0
    
    # Start with random ±1 assignment
    rng = np.random.default_rng(12345)
    best_h = rng.choice([-1.0, 1.0], size=N)
    best_val = _objective(best_h)
    
    current_h = best_h.copy()
    current_val = best_val
    
    # Simulated annealing
    T = 0.1
    T_min = 0.001
    cooling = 0.9995
    
    while time.time() < deadline:
        idx = rng.integers(0, N)
        old_h = current_h[idx]
        current_h[idx] = -current_h[idx]
        new_val = _objective(current_h)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = current_h.copy()
        else:
            delta = new_val - current_val
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_val = new_val
            else:
                current_h[idx] = old_h
        
        T = max(T_min, T * cooling)
        
        if T <= T_min * 1.01 and rng.random() < 0.05:
            current_h = rng.choice([-1.0, 1.0], size=N)
            current_val = _objective(current_h)
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
            T = 0.1
    
    return best_h
