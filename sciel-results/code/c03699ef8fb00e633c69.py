import numpy as np

N = 400
w = 1.0 / 200.0  # interval width

def construct():
    # Initialize with a random binary sequence
    s = np.random.choice([-1, 1], size=N)
    
    def get_autocorr_max(s):
        # Compute autocorrelation using FFT
        # r[k] = sum_{i} s[i] s[i+k]
        # This is the circular autocorrelation if we use FFT directly on N points.
        # We need linear autocorrelation.
        # We can zero-pad to 2*N to avoid circular wrap-around.
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        return np.max(np.abs(r)) * w

    best_s = s.copy()
    best_val = get_autocorr_max(s)
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-6
    iterations = 20000
    
    current_val = best_val
    
    for i in range(iterations):
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Flip the sign
        s[idx] *= -1
        
        # Calculate new objective
        new_val = get_autocorr_max(s)
        
        # Decide whether to accept the move
        delta = new_val - current_val
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_s = s.copy()
        else:
            # Reject: flip back
            s[idx] *= -1
            
        T *= cooling_rate
        if T < min_T:
            break
            
    return best_s
