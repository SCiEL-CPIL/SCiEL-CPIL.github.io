"""Improved program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400
DEADLINE = 145  # safety margin before 150s kill


def _max_autocorr_abs(s):
    """Compute max |autocorr| over non-zero lags for a length-N sequence.
    
    Returns the maximum of |r[k]| for k=1..N-1 where r is the linear autocorrelation.
    """
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # Linear autocorrelation: acf[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k]
    # Normalize: divide by N
    acf = acf / N
    return float(np.max(np.abs(acf[1:])))


def _score(s):
    """Upper bound on C3; lower is better."""
    return _max_autocorr_abs(s)

def _chirp_sequence(N):
    """Generate a real chirp-like sequence."""
    # Use a quadratic phase
    n = np.arange(N)
    # Scale to fit in [0, 2*pi] or similar
    # For ±1, we can use sign of cos
    phase = np.pi * n * n / N
    return np.sign(np.cos(phase)).astype(np.int8)

def construct():
    t_start = time.time()
    
    # Seed value: all ones
    seed_val = _score(np.ones(N, dtype=np.int8))
    
    # Initialize with Chirp
    best_seq = _chirp_sequence(N)
    best_score = _score(best_seq)
    
    # Also check ones just in case
    ones_score = seed_val
    if ones_score < best_score:
        best_seq = np.ones(N, dtype=np.int8)
        best_score = ones_score
        
    rng = np.random.default_rng(42)
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-5
    
    current_seq = best_seq.copy()
    current_score = best_score
    
    while T > min_T:
        if time.time() - t_start > DEADLINE:
            break
        
        # Pick a random position to flip
        i = rng.integers(N)
        old_val = current_seq[i]
        current_seq[i] = -old_val
        new_score = _score(current_seq)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_seq = current_seq.copy()
        else:
            current_seq[i] = old_val
            
        T *= cooling_rate
        
    return best_seq
