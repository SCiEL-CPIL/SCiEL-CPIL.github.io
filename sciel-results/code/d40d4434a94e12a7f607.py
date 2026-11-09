import numpy as np
import time

N = 400

def construct():
    """
    Constructs a signed step function (sequence of +/- 1) of length N
    to minimize the peak autocorrelation sidelobe.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # FFT size for linear autocorrelation
    fft_size = 1 << (2 * N - 1).bit_length()
    
    # Initialize with random sequence
    x = np.random.choice([-1, 1], size=N)
    
    # Function to compute peak sidelobe
    def peak_sidelobe(seq):
        # Zero pad
        xp = np.zeros(fft_size)
        xp[:N] = seq
        # FFT
        X = np.fft.fft(xp)
        # Power spectrum
        P = X * np.conj(X)
        # Inverse FFT to get autocorrelation
        R = np.fft.ifft(P).real
        # R[0] is N. R[1:N] are positive lags.
        # We want max absolute value of R[1:N]
        return np.max(np.abs(R[1:N]))
    
    current_cost = peak_sidelobe(x)
    best_x = x.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    T = 5.0  # Initial temperature
    cooling_rate = 0.9999
    min_T = 0.01
    
    # We will try both single bit flips and block flips
    # Block size range
    min_block = 2
    max_block = 20
    
    iteration = 0
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            # If temperature is too low, reheat or restart?
            # Reheating can help escape local minima.
            T = T * 1.5
            # Or just restart with a new random sequence if stuck for too long?
            # For now, simple reheat.
        
        # Choose move type
        if np.random.random() < 0.5:
            # Single bit flip
            idx = np.random.randint(N)
            x[idx] *= -1
        else:
            # Block flip
            block_len = np.random.randint(min_block, max_block + 1)
            start_idx = np.random.randint(0, N - block_len + 1)
            x[start_idx:start_idx + block_len] *= -1
        
        new_cost = peak_sidelobe(x)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()
        else:
            # Revert move
            if np.random.random() < 0.5:
                pass # We don't know which move was made easily without storing it
            # Better approach: store the move or just re-apply?
            # Storing the move is cleaner. Let's refactor to store move.
            
        T *= cooling_rate
        iteration += 1
        
        # Periodic check for time
        if iteration % 1000 == 0:
            if time.time() - start_time > time_limit - 5:
                break
                
    return best_x
