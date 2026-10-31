import numpy as np
from scipy.signal import windows
import time

N = 400

def _get_max_sidelobe(f):
    """Compute the maximum absolute value of the autoconvolution sidelobes."""
    # Use FFT for efficient autoconvolution
    # Zero-pad to avoid circular convolution artifacts
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    # The autoconvolution is centered at N//2 in the FFT output if we don't shift, 
    # but standard linear convolution via FFT:
    # Let's just use np.convolve for simplicity and correctness, N=400 is small.
    ac = np.convolve(f, f, mode='full')
    # The peak is at index N-1 (for zero-based indexing of full convolution of length N)
    # Sidelobes are everything else.
    # We want max(abs(ac)) excluding the main lobe.
    # The main lobe is at the center.
    center = N - 1
    # To be safe, exclude a small window around the center? 
    # The task says "upper bound on C3", likely max sidelobe.
    # For a step function, the "sidelobes" are k != 0.
    # In the 'full' convolution, index i corresponds to lag i - (N-1).
    # Lag 0 is at index N-1.
    # We check all indices except N-1.
    sidelobes = np.delete(ac, center)
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_beta = 8.0
    best_val = float('inf')
    
    # Search for the best Kaiser window beta
    # Kaiser window is a good approximation to the optimal window for sidelobe minimization
    for beta in np.arange(0, 20, 0.5):
        if time.time() > deadline:
            break
        # Generate Kaiser window
        # scipy.signal.windows.kaiser expects length and beta
        f = windows.kaiser(N, beta)
        # Normalize to L2 norm 1
        norm = np.linalg.norm(f)
        if norm == 0:
            continue
        f = f / norm
        
        val = _get_max_sidelobe(f)
        if val < best_val:
            best_val = val
            best_beta = beta
            
    # Generate the final window with the best beta
    f = windows.kaiser(N, best_beta)
    norm = np.linalg.norm(f)
    if norm == 0:
        # Fallback to ones if something went wrong
        return np.ones(N)
    f = f / norm
    
    return f
