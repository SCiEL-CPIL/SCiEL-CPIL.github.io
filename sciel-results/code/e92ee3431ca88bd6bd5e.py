import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with random +/- 1
    x = np.random.choice([-1, 1], size=N).astype(np.int8)
    
    # Function to compute autocorrelation vector c[0..N-1]
    # c[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
    def compute_autocorr(x):
        # Zero pad to 2N for linear autocorrelation
        X = np.fft.fft(x, n=2*N)
        C = np.fft.ifft(X * X)
        c = C[:N].real
        return c

    # Initial autocorrelation
    c = compute_autocorr(x)
    best_psl = np.max(np.abs(c[1:]))
    best_x = x.copy()
    best_c = c.copy()
    
    # Local Search with Incremental Update
    # We maintain c and update it incrementally
    # c[k] for k in 1..N-1
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Pre-allocate temp array for deltas? Or just update in place and revert?
    # Updating in place is risky if we need to revert.
    # Strategy: 
    # 1. Calculate new_c = c.copy()
    # 2. Apply updates
    # 3. Check max
    # 4. If accept, x[i] = -x[i], c = new_c
    # 5. Else, discard new_c
    
    # To speed up, we can avoid full copy if we store the deltas.
    # But N=400 is small. Copying 400 floats is fast.
    
    while time.time() < deadline:
        i = np.random.randint(N)
        
        # Calculate changes to c
        # We will create a copy of c to test
        new_c = c.copy()
        
        old_val = x[i]
        
        # Update lags k where i is the left index: i+k < N
        # k ranges from 1 to N-1-i
        # new_c[k] -= 2 * old_val * x[i+k]
        # Vectorized update for the right side
        if N - 1 - i > 0:
            k_right = np.arange(1, N - i)
            # x[i+k_right]
            new_c[k_right] -= 2 * old_val * x[i + k_right]
            
        # Update lags k where i is the right index: i-k >= 0
        # k ranges from 1 to i
        # new_c[k] -= 2 * x[i-k] * old_val
        if i > 0:
            k_left = np.arange(1, i + 1)
            # x[i-k_left]
            new_c[k_left] -= 2 * x[i - k_left] * old_val
            
        new_psl = np.max(np.abs(new_c[1:]))
        
        delta = new_psl - best_psl
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            x[i] = -old_val
            c = new_c
            best_psl = new_psl
            best_x = x.copy()
            best_c = c.copy() # Keep best_c consistent? Not strictly needed if we track best_psl, but good for restarts if we had any.
        else:
            # Reject: x and c remain unchanged
            pass
            
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_x
