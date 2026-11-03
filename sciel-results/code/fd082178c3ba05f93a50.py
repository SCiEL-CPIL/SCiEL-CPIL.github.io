import numpy as np
import time

N = 600

def _compute_peak_sidelobe(h):
    """
    Computes the maximum value of the linear autocorrelation 
    for non-zero lags.
    """
    # Zero pad to 2*N to handle linear convolution via FFT
    h_padded = np.zeros(2 * N)
    h_padded[:N] = h
    
    # Compute FFT
    H = np.fft.fft(h_padded)
    # Power spectrum
    P = H * np.conj(H)
    # Inverse FFT to get autocorrelation
    acf = np.fft.ifft(P).real
    
    # acf[0] is lag 0
    # acf[1:N] corresponds to lags 1 to N-1
    # We ignore negative lags due to symmetry for real signals
    if N <= 1:
        return 0.0
    return np.max(acf[1:N])

def construct():
    start_time = time.time()
    # Allow 110 seconds for computation to be safe
    deadline = start_time + 110.0
    
    # Initialize with a random 0/1 sequence
    np.random.seed(123)
    h = (np.random.rand(N) < 0.5).astype(float)
    
    current_peak = _compute_peak_sidelobe(h)
    best_h = h.copy()
    best_peak = current_peak
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.05
    alpha = 0.995
    
    # Main optimization loop
    while time.time() < deadline:
        # Select a random index to flip
        i = np.random.randint(N)
        
        # Flip the bit
        h[i] = 1.0 - h[i]
        
        # Evaluate new peak sidelobe
        new_peak = _compute_peak_sidelobe(h)
        
        # Calculate delta
        delta = new_peak - current_peak
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_h = h.copy()
        else:
            # Revert the flip
            h[i] = 1.0 - h[i]
            
        # Cool down
        T *= alpha
        
    return best_h.tolist()
