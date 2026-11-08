import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 20s margin
    
    # Initialize with a random sequence of +1 and -1
    # Using a seed for reproducibility in local testing, but random is fine for evaluator
    x = np.random.choice([-1, 1], size=N)
    
    # Function to compute max absolute autocorrelation for k != 0
    def max_sidelobe(x):
        # Use FFT for efficient autocorrelation
        # Zero-pad to 2*N to avoid circular convolution effects
        n = len(x)
        f = np.fft.fft(x, n=2*n)
        acf = np.fft.ifft(f * np.conj(f)).real
        # acf[0] is c_0. We want max of acf[k] for k != 0
        # Note: acf is symmetric. acf[1:N] are the unique lags.
        # acf[N+1:2N] are the negative lags.
        # We take max of absolute values.
        return np.max(np.abs(acf[1:n]))
    
    # Initial score
    current_score = max_sidelobe(x)
    best_x = x.copy()
    best_score = current_score
    
    # Local Search: Greedy sign flipping
    # Iterate until time limit or no improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # To speed up, we can sample indices or iterate all
        # For N=400, iterating all is 400 steps.
        # Each step requires FFT of size 800. ~10us. 400*10us = 4ms per pass.
        # We can do many passes.
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Flip sign at i
            x[i] = -x[i]
            
            new_score = max_sidelobe(x)
            
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
                improved = True
            else:
                # Revert
                x[i] = -x[i]
                
    return best_x
# EVOLVE-BLOCK-END
