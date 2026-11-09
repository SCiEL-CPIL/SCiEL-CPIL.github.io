"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_periodic(h):
    """Compute periodic autocorrelation of a ±1 sequence using FFT."""
    f = np.fft.fft(h.astype(float))
    power = f * np.conj(f)
    acorr = np.fft.ifft(power).real
    return acorr


def _local_search(h, R, deadline):
    """Perform local search with incremental autocorrelation updates."""
    best_score = np.max(np.abs(R[1:]))
    best_h = h.copy()
    h = h.copy()
    R = R.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for j in range(N):
            s = h[j]
            idx_plus = (j + np.arange(N)) % N
            idx_minus = (j - np.arange(N)) % N
            delta = -2.0 * s * (h[idx_plus] + h[idx_minus])
            new_R = R + delta
            new_score = np.max(np.abs(new_R[1:]))
            
            if new_score < best_score - 1e-12:
                h[j] = -s
                R = new_R
                best_score = new_score
                best_h = h.copy()
                improved = True
            
            if time.time() > deadline:
                break
    
    return best_h, best_score


def construct():
    deadline = time.time() + 140
    
    np.random.seed(42)
    best_h = None
    best_score = float('inf')
    
    # Try multiple random starts with local search
    num_starts = 30
    for start in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        R = _autocorr_periodic(h)
        
        h_opt, score = _local_search(h, R, deadline)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
