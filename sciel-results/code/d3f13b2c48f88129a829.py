import numpy as np
import time

N = 400

def _peak_autocorr_fft(s):
    """Compute the peak absolute value of aperiodic autocorrelation for non-zero lags using FFT."""
    # FFT size must be at least 2*N to avoid circular convolution aliasing
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    H = np.fft.rfft(s, n=fft_size)
    # Power spectrum
    power = np.abs(H) ** 2
    # Inverse FFT gives the circular autocorrelation
    acf = np.fft.irfft(power, n=fft_size)
    
    # The aperiodic autocorrelation for lag k is acf[k] for k=0..N-1
    # and acf[fft_size-k] for k=1..N-1 (negative lags)
    # We want max absolute value for k != 0
    # acf[0] is zero lag
    # lags 1 to N-1 are in acf[1:N]
    # lags -1 to -(N-1) are in acf[fft_size-N+1:fft_size]
    
    # Due to symmetry for real signals, acf[k] == acf[-k]
    # So we just need to check positive lags 1 to N-1
    return np.max(np.abs(acf[1:N]))

def construct():
    start_time = time.time()
    timeout = start_time + 110.0  # Safety margin
    
    # Initializations
    # 1. Thue-Morse
    tm_seq = np.array([bin(i).count('1') % 2 for i in range(N)])
    s_tm = 1 - 2 * tm_seq
    
    # 2. Random
    rng = np.random.default_rng(42)
    s_rand = rng.choice([-1, 1], size=N)
    
    candidates = [s_tm, s_rand]
    
    best_s = None
    best_peak = np.inf
    
    for s_init in candidates:
        s = s_init.copy()
        current_peak = _peak_autocorr_fft(s)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_s = s.copy()
            
        # Simulated Annealing
        T = 10.0  # Initial temperature
        cooling_rate = 0.99
        min_T = 0.1
        
        while T > min_T and time.time() < timeout:
            improved = False
            # Try a number of flips per temperature step
            for _ in range(100):
                if time.time() > timeout:
                    break
                i = rng.integers(0, N)
                old_val = s[i]
                s[i] = -old_val
                new_peak = _peak_autocorr_fft(s)
                
                delta = new_peak - current_peak
                # Accept if better, or with probability exp(-delta/T)
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_s = s.copy()
                    improved = True
                else:
                    s[i] = old_val
            
            T *= cooling_rate
            
    return best_s
