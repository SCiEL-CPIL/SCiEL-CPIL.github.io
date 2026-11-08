"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(x):
    """Compute the score: max|autoconvolution| / (sum x)^2"""
    if np.sum(x) == 0:
        return np.inf
    # Compute autoconvolution
    ac = np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    max_abs_ac = np.max(np.abs(ac))
    sum_x = np.sum(x)
    return max_abs_ac / (sum_x ** 2)


def construct():
    """
    Construct a band-limited signal with random phases to minimize
    the maximum absolute autoconvolution relative to squared integral.
    """
    best_x = None
    best_score = np.inf
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Try different bandwidths
    bandwidths = [1, 2, 5, 10, 20, 50, 100, 200]
    
    # Number of trials per bandwidth
    trials_per_bw = 50
    
    for M in bandwidths:
        if time.time() - start_time > time_limit:
            break
        
        # Determine which frequency bins to activate
        # For real signals, we need conjugate symmetry
        # Activate bins: 0, 1, ..., M-1 and N-M+1, ..., N-1
        # (with special handling for DC and Nyquist if N is even)
        
        for trial in range(trials_per_bw):
            if time.time() - start_time > time_limit:
                break
            
            X = np.zeros(N, dtype=complex)
            
            # DC component (always real)
            X[0] = 1.0
            
            # Positive frequency bins
            for k in range(1, min(M, N//2)):
                phase = 2 * np.pi * np.random.random()
                amp = 1.0  # Unit amplitude
                X[k] = amp * np.exp(1j * phase)
                # Conjugate symmetry
                X[N-k] = np.conj(X[k])
            
            # Handle Nyquist bin if N is even and M > N//2
            if N % 2 == 0 and M > N//2:
                # Nyquist bin must be real
                X[N//2] = 1.0 if np.random.random() < 0.5 else -1.0
            
            # Inverse DFT to get time-domain signal
            x = np.fft.ifft(X).real
            
            # Check if signal is reasonable (not all zeros)
            if np.max(np.abs(x)) < 1e-10:
                continue
            
            # Compute score
            score = _compute_score(x)
            
            if score < best_score:
                best_score = score
                best_x = x.copy()
    
    # If no good signal found, fall back to constant signal
    if best_x is None:
        x = np.ones(N) / N
    else:
        x = best_x
    
    return x
# EVOLVE-BLOCK-END
