"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr(f):
    """Compute aperiodic autocorrelation for all lags."""
    n = len(f)
    # Use FFT for linear autocorrelation
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad, size)
    acf = np.fft.ifft(F * np.conj(F)).real
    # acf[k] for k=0..n-1 is sum_i f[i]*f[i+k] (with appropriate wrapping handled by zero-padding)
    # Actually with zero-padding to 2n, acf[k] = sum_{i=0}^{n-1-k} f[i]*f[i+k] for k=0..n-1
    # And acf[n+k] = sum_{i=k}^{n-1} f[i]*f[i-k] for k=0..n-1
    # We want max |acf[k]| for k != 0
    if len(acf) > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0


def _autocorr_values(f):
    """Return all autocorrelation values for non-zero lags."""
    n = len(f)
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad, size)
    acf = np.fft.ifft(F * np.conj(F)).real
    if len(acf) > 1:
        return acf[1:]
    return np.array([])


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Start with random ±1 sequence
    rng = np.random.RandomState(42)
    f = rng.choice([-1.0, 1.0], size=N)
    
    best_f = f.copy()
    best_score = _autocorr(f)
    
    # Local search: try flipping individual bits
    # Focus on reducing the maximum autocorrelation
    improved = True
    while improved and time.time() < deadline:
        improved = False
        acf_vals = _autocorr_values(f)
        current_max = np.max(np.abs(acf_vals)) if len(acf_vals) > 0 else 0.0
        
        # Find the worst lag
        if len(acf_vals) > 0:
            worst_idx = np.argmax(np.abs(acf_vals))
        
        # Try flipping each bit and see if it improves
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = f[i]
            f[i] = -old_val
            
            new_score = _autocorr(f)
            
            if new_score < current_max:
                current_max = new_score
                improved = True
            else:
                f[i] = old_val
        
        # Update best
        if current_max < best_score:
            best_score = current_max
            best_f = f.copy()
    
    return best_f
# EVOLVE-BLOCK-END
