import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Use a size that is a power of 2 for FFT efficiency if possible, 
    # but 2*N = 800 is fine. Next power of 2 is 1024.
    fft_size = 1024
    
    def compute_objective(x):
        # x is array of 1, -1
        # Zero pad to fft_size
        # Using rfft for real input
        F = np.fft.rfft(x, fft_size)
        P = F * np.conj(F)
        R = np.fft.irfft(P, fft_size)
        # R[0:N] are the autocorrelation lags 0 to N-1
        # We want lags 1 to N-1
        side_lobe = R[1:N]
        # Metric: Sum of cubes of absolute values
        return np.sum(np.abs(side_lobe) ** 3)

    # Initialize with random +/- 1
    x = np.random.choice([1, -1], size=N)
    
    current_obj = compute_objective(x)
    best_x = x.copy()
    best_obj = current_obj
    
    # Simulated Annealing parameters
    T = 100.0
    T_min = 0.01
    alpha = 0.9995
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Flip
        old_val = x[i]
        x[i] = -old_val
        
        new_obj = compute_objective(x)
        
        delta = new_obj - current_obj
        
        # Metropolis criterion
        if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
            # Accept
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = x.copy()
        else:
            # Reject
            x[i] = old_val
            
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_x

# EVOLVE-BLOCK-END
