"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3_bound(f):
    """
    Compute the autoconvolution upper bound C3 for a signed step function.
    
    The autocorrelation R(k) = sum_i f[i] * f[i+k] (circular or linear).
    C3 is the maximum |R(k)| for k != 0, which gives an upper bound.
    We use linear autocorrelation (zero-padded) to be safe.
    """
    # Linear autocorrelation via FFT for efficiency
    n = len(f)
    # Zero-pad to avoid circular convolution
    size = 2 * n
    F = np.fft.rfft(f, size)
    R = np.fft.irfft(F * np.conj(F), size)
    # R[k] for k=0,...,n-1 is the autocorrelation at lag k
    # R[n-k] gives negative lags
    # We want max |R[k]| for k != 0
    # R[0] is the DC (lag 0), skip it
    # For linear autocorrelation, lags 1 to n-1 are in R[1:n]
    # Negative lags are in R[n+1:2n] but by symmetry R[-k] = R[k] for real signals
    # So just check R[1:n]
    if n > 1:
        peak = np.max(np.abs(R[1:n]))
    else:
        peak = 0.0
    return peak


def _evaluate(f):
    """Compute the score (lower is better)."""
    return _compute_c3_bound(f)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initialize with random signed step function
    rng = np.random.default_rng(42)
    f = rng.choice([-1.0, 1.0], size=N)
    
    best_f = f.copy()
    best_score = _evaluate(best_f)
    
    # Local search: flip individual bits
    # Try to find improvements by flipping one element at a time
    improved = True
    iteration = 0
    
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        
        # Try flipping each position
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            
            # Flip position i
            f[i] = -f[i]
            score = _evaluate(f)
            
            if score < best_score:
                best_score = score
                best_f = f.copy()
                improved = True
            else:
                # Revert
                f[i] = -f[i]
        
        # If no improvement in this pass, try random restarts or perturbations
        if not improved:
            # Try a few random perturbations (flip multiple bits)
            for restart in range(5):
                if time.time() > deadline - 5.0:
                    break
                
                f = best_f.copy()
                # Flip a random subset of positions
                n_flips = rng.integers(1, 10)
                indices = rng.choice(N, size=n_flips, replace=False)
                f[indices] = -f[indices]
                score = _evaluate(f)
                
                if score < best_score:
                    best_score = score
                    best_f = f.copy()
                    improved = True
    
    return best_f
# EVOLVE-BLOCK-END
