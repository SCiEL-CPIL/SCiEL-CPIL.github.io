"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_peak(x):
    """
    Compute max absolute value of autoconvolution of x, excluding zero-lag.
    Uses FFT for efficiency.
    """
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    f = np.fft.rfft(x, n_fft)
    # Power spectrum
    power = f * np.conj(f)
    # Inverse FFT to get autocorrelation
    conv = np.fft.irfft(power, n_fft)
    # The autocorrelation is centered. 
    # For linear autocorrelation of length N, the relevant lags are 0 to N-1 
    # and N-1 down to 1 (symmetric).
    # conv[0] is lag N (due to circular shift in FFT output interpretation for linear conv padded)
    # Actually, np.fft.irfft of power spectrum gives circular autocorrelation.
    # Since we padded to 2N, the first N bins (0 to N-1) correspond to lags 0 to N-1?
    # No, for linear convolution via FFT with padding 2N:
    # The result is linear convolution.
    # conv[0] is sum x[i]x[i] = Energy (Lag 0).
    # conv[1] is sum x[i]x[i+1] (Lag 1).
    # ...
    # conv[N-1] is sum x[i]x[i+N-1] (Lag N-1).
    # The rest are lags > N-1 which are 0 for linear conv, but due to circular nature of FFT
    # without proper windowing, we might see wrap-around if not careful.
    # However, with padding 2N, the linear conv fits in the first N bins (lags 0 to N-1) 
    # and the symmetry is in the second half.
    # Specifically, conv[k] for k=0..N-1 are the positive lags.
    # We want max abs for k=1..N-1.
    
    # Extract lags 1 to N-1
    return np.max(np.abs(conv[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Strategy: Dense +/-1 sequence optimization using Simulated Annealing / Hill Climbing.
    # This addresses the "sparse fails" observation by using full energy.
    
    best_x = np.random.choice([-1, 1], size=N)
    best_score = _autoconv_peak(best_x)
    
    # Precompute FFT size
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
        
    def _get_score(x):
        f = np.fft.rfft(x, n_fft)
        conv = np.fft.irfft(f * np.conj(f), n_fft)
        return np.max(np.abs(conv[1:N]))

    # Simulated Annealing
    current_x = best_x.copy()
    current_score = best_score
    temperature = 10.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        # Try a random flip
        idx = np.random.randint(N)
        current_x[idx] *= -1
        
        new_score = _get_score(current_x)
        
        # Accept if better, or with probability exp(-(delta)/T) if worse
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if np.random.random() < np.exp(-delta / temperature):
                current_score = new_score
            else:
                current_x[idx] *= -1 # Revert
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1
            
    return best_x
# EVOLVE-BLOCK-END
