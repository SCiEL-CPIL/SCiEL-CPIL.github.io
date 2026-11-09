"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

N_FFT = 1 << (2 * N).bit_length()

def _autocorr_peak(h):
    """Compute max absolute autocorrelation sidelobe using FFT."""
    F = np.fft.rfft(h, n=N_FFT)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=N_FFT)
    return np.max(np.abs(r[1:N]))

def _simulated_annealing(h, deadline, T_start=10.0, T_end=0.01, total_steps=500000):
    """
    Perform simulated annealing on the bit sequence h.
    Returns the best h and its peak value found.
    """
    current_h = h.copy()
    current_peak = _autocorr_peak(current_h)
    best_h = current_h.copy()
    best_peak = current_peak
    
    cooling_rate = (T_end / T_start) ** (1.0 / total_steps)
    T = T_start
    rng = np.random.default_rng(42)
    
    for step in range(total_steps):
        if time.time() > deadline:
            break
            
        i = rng.integers(N)
        current_h[i] = -current_h[i]
        
        new_peak = _autocorr_peak(current_h)
        
        if new_peak < current_peak:
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_h = current_h.copy()
        else:
            delta = new_peak - current_peak
            if delta > 1e-9:
                if T > 1e-9:
                    p = math.exp(-delta / T)
                    if rng.random() < p:
                        current_peak = new_peak
                    else:
                        current_h[i] = -current_h[i]
                else:
                    current_h[i] = -current_h[i]
            else:
                current_peak = new_peak
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_h = current_h.copy()
        
        T *= cooling_rate
        
    return best_h, best_peak

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(123)
    h = rng.choice([-1.0, 1.0], size=N)
    
    best_h, best_peak = _simulated_annealing(h, deadline)
    
    # If we have significant time left, try another independent run
    if time.time() < deadline - 50.0:
        h2 = rng.choice([-1.0, 1.0], size=N)
        h2, peak2 = _simulated_annealing(h2, deadline)
        if peak2 < best_peak:
            best_h = h2
            best_peak = peak2
            
    return best_h
# EVOLVE-BLOCK-END
