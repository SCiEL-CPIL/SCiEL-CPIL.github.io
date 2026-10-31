"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(h):
    """Compute max absolute value of full autoconvolution."""
    # Use FFT for efficiency
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size=size)
    ac = np.fft.irfft(f * np.conj(f), size=size)[:n]
    return np.max(np.abs(ac))


def _score(h):
    """Compute the objective: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    return _autoconv_max_abs(h) / (s * s)


def construct():
    """Construct signed step function using low-discrepancy Kronecker sequence."""
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s limit
    
    # Start with a Kronecker sequence based on sqrt(2)
    # h[i] = sign of the fractional part distance to 0.5
    # This creates a balanced +/-1 sequence with low discrepancy
    alpha = np.sqrt(2.0)  # irrational for good distribution
    i = np.arange(N, dtype=np.float64)
    frac = (i * alpha) % 1.0
    # Map to signs: +1 if frac < 0.5, -1 otherwise
    h = np.where(frac < 0.5, 1.0, -1.0)
    
    # Ensure the sum is not zero (adjust if needed)
    s = np.sum(h)
    if abs(s) < 1.0:
        # Flip the element with largest index to increase sum
        h[-1] = 1.0
        s = np.sum(h)
    
    best_h = h.copy()
    best_score = _score(best_h)
    
    # Local search: try flipping individual signs
    # Use a greedy hill-climbing approach
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            h[idx] = -h[idx]
            new_score = _score(h)
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                h[idx] = -h[idx]  # revert
    
    # Try multiple different alpha values for diversity
    alphas = [np.sqrt(2.0), np.sqrt(3.0), np.sqrt(5.0), (1 + np.sqrt(5.0))/2.0]
    for alpha in alphas:
        if time.time() > deadline:
            break
        frac = (i * alpha) % 1.0
        h = np.where(frac < 0.5, 1.0, -1.0)
        s = np.sum(h)
        if abs(s) < 1.0:
            h[-1] = 1.0
        
        sc = _score(h)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
        
        # Local search from this starting point
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                h[idx] = -h[idx]
                new_score = _score(h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
                    improved = True
                else:
                    h[idx] = -h[idx]
    
    return best_h
# EVOLVE-BLOCK-END
