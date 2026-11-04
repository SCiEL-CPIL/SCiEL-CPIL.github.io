"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr(h):
    """Compute autocorrelation for all lags 0..N-1."""
    # A[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k]
    # Use FFT for speed
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(h, size=size)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, size=size)[:n]
    return acf


def _score(h):
    """Return the upper bound on C3 (max autocorrelation)."""
    acf = _autocorr(h)
    return float(np.max(np.abs(acf)))


def _random_signed(n, seed):
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=n)


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s limit
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try several random initializations
    for seed in range(50):
        if time.time() > deadline:
            break
        h = _random_signed(N, seed)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Local search: flip individual bits
    h = best_h.copy()
    current_score = best_score
    improved = True
    max_iter = 200000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Try random flips
        rng = np.random.RandomState(it + 1000)
        for _ in range(20):
            if time.time() > deadline:
                break
            idx = rng.randint(N)
            h[idx] *= -1
            s = _score(h)
            if s < current_score - 1e-12:
                current_score = s
                improved = True
            else:
                h[idx] *= -1  # revert
    
    # Final high-precision polish: try all single-bit flips
    if time.time() < deadline - 5:
        for idx in range(N):
            if time.time() > deadline:
                break
            h[idx] *= -1
            s = _score(h)
            if s < current_score - 1e-12:
                current_score = s
            else:
                h[idx] *= -1
    
    return h
# EVOLVE-BLOCK-END
