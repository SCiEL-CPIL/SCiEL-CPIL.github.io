import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_x = None
    best_score = float('inf')
    
    # Helper to compute max abs autoconvolution
    def max_abs_autoconv(x):
        # Use FFT for O(N log N)
        # Zero padding to 2*N to avoid circular convolution issues for linear autoconv
        # Actually, for autoconvolution of length N, the result length is 2N-1.
        # FFT size should be at least 2N-1. Let's use next power of 2 for speed.
        fft_size = 1 << (2 * N - 1).bit_length()
        X = np.fft.fft(x, n=fft_size)
        A = np.fft.ifft(X * X).real
        return np.max(np.abs(A))
    
    # Try a few random initializations
    num_restarts = 5
    for r in range(num_restarts):
        x = np.random.choice([1, -1], N)
        current_score = max_abs_autoconv(x)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
        # Local search
        # We will perform a fixed number of steps or until time runs out
        # Let's do a budget of steps, but check time frequently
        steps_in_round = 5000
        for step in range(steps_in_round):
            if time.time() > deadline:
                break
            
            # Pick a random index to flip
            i = np.random.randint(N)
            old_val = x[i]
            x[i] = -old_val
            
            new_score = max_abs_autoconv(x)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                # Revert
                x[i] = old_val
        
        if time.time() > deadline:
            break
            
    return best_x
