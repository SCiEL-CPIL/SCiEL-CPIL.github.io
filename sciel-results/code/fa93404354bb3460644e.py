"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate the Thue-Morse sequence of length n mapped to {-1, 1}."""
    # t_n is the parity of the number of 1s in the binary representation of n
    # We map 0 -> 1, 1 -> -1
    t = np.zeros(n, dtype=int)
    for i in range(n):
        # Efficient way: t[i] = t[i//2] ^ (i%2)
        if i > 0:
            t[i] = t[i // 2] ^ (i % 2)
        else:
            t[i] = 0
    # Map to signed values
    return 1.0 - 2.0 * t

def _compute_autoconvolution_peak(f):
    """
    Compute the maximum absolute value of the autoconvolution (autocorrelation)
    at non-zero lags.
    """
    n = len(f)
    if n <= 1:
        return 0.0
    
    # Use FFT to compute autocorrelation
    # Size should be at least 2*n - 1 for linear autocorrelation
    # Next power of 2 for efficiency
    size = 1
    while size < 2 * n:
        size *= 2
        
    F = np.fft.rfft(f, n=size)
    # Power spectrum
    power = F * np.conj(F)
    # Inverse FFT gives the autocorrelation
    acf = np.fft.irfft(power, n=size).real
    
    # acf[0] is the zero-lag (energy). We ignore it.
    # For real signals, acf is symmetric: acf[k] = acf[-k].
    # We only need to check positive lags 1 to n-1.
    # Note: due to circular convolution in FFT, lags > n/2 wrap around.
    # But since we zero-padded to size >= 2n, the linear autocorrelation is correct for lags 0..n-1.
    
    # Extract non-zero lags
    if n > 1:
        return float(np.max(np.abs(acf[1:n])))
    return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialize with Thue-Morse sequence
    # This provides a structurally low-peak autocorrelation profile
    f = _thue_morse(N)
    current_score = _compute_autoconvolution_peak(f)
    
    best_f = f.copy()
    best_score = current_score
    
    # 2. Local Search Refinement
    # We perform multiple passes of sign flipping.
    # In each pass, we iterate through indices and flip if it improves the score.
    
    passes = 0
    while time.time() < deadline:
        improved = False
        passes += 1
        
        # Randomize the order of indices to escape local minima cycles
        # Using a fixed seed per pass index might be better for reproducibility, 
        # but random is fine for search.
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try flipping sign at i
            old_val = f[i]
            f[i] = -old_val
            
            new_score = _compute_autoconvolution_peak(f)
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                # Revert
                f[i] = old_val
        
        if not improved:
            break
            
        # Update best if current is better
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
            
    return best_f
# EVOLVE-BLOCK-END
