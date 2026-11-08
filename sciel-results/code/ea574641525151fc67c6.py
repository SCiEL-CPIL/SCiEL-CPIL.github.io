import numpy as np
import time

N = 400
DEADLINE = time.time() + 140

def _score(h):
    s = np.sum(h)
    energy = np.sum(h**2)
    if energy < 1e-12:
        return float('inf')
    autocorr = np.correlate(h, h, mode='full')
    # Exclude lag 0 (which is at index N-1 in 'full' mode)
    # autocorr has length 2N-1, lag 0 is at index N-1
    sidelobes = np.concatenate([autocorr[:N-1], autocorr[N:]])
    max_sidelobe = np.max(np.abs(sidelobes))
    return max_sidelobe / energy

def construct():
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    start = time.time()
    rng = np.random.default_rng(42)
    
    # Try random ±1 sequences
    for trial in range(5000):
        if time.time() > DEADLINE:
            break
        h = rng.choice([-1, 1], N)
        sc = _score(h)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
    
    # Local search: flip random bits
    for trial in range(10000):
        if time.time() > DEADLINE:
            break
        h = best_h.copy()
        idx = rng.integers(0, N)
        h[idx] *= -1
        sc = _score(h)
        if sc < best_score:
            best_score = sc
            best_h = h
    
    return best_h
