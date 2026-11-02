import numpy as np
import time

N = 400

def _compute_autocorr(s):
    """
    Compute the aperiodic autocorrelation of s.
    Returns array of length N, where index k is sum_{n=0}^{N-1-k} s[n]s[n+k].
    """
    # Use FFT for speed
    # Pad to 2N to avoid circular convolution effects
    n_fft = 1 << (2 * N - 1).bit_length()
    S = np.fft.fft(s, n_fft)
    P = S * np.conj(S)
    A = np.fft.ifft(P).real
    return A[:N]

def _max_abs_autocorr(s):
    A = _compute_autocorr(s)
    # Exclude lag 0
    return np.max(np.abs(A[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Initialize with random sequence
    # Use a fixed seed for reproducibility? No, let it vary or use a good heuristic.
    # Random is a good start.
    s = np.random.choice([-1, 1], N)
    
    best_s = s.copy()
    best_val = _max_abs_autocorr(best_s)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.995
    steps_per_iter = 100
    
    # Precompute indices for fast updates? 
    # For N=400, FFT is fast enough (~0.1ms). 
    # 100 steps * 10000 iters = 1,000,000 evaluations. 
    # 1,000,000 * 0.1ms = 100s. Too slow.
    # Need faster local search or fewer steps.
    
    # Better: Greedy Hill Climbing with random restarts.
    # Or, update only the affected lags.
    
    # Let's try a faster local search:
    # 1. Compute A.
    # 2. Find k_max with max |A_k|.
    # 3. Try flipping s_i for i in range(N).
    #    Delta A_k for flip at i:
    #    A_k changes by -2 s_i s_{i+k} (if i+k < N)
    #    A_{k'} changes for other lags.
    #    This is complex.
    
    # Alternative: Use the FFT method but optimize the loop.
    # N=400 is small.
    # Let's estimate FFT cost.
    # np.fft.fft on size 1024 is very fast.
    
    # Let's run a loop until deadline.
    
    current_s = s.copy()
    current_val = best_val
    
    while time.time() < deadline:
        # Try random flips
        # Pick a random index
        i = np.random.randint(N)
        
        # Flip
        current_s[i] *= -1
        
        # Evaluate
        new_val = _max_abs_autocorr(current_s)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = current_s.copy()
        else:
            # Revert
            current_s[i] *= -1
            # Simple acceptance probability for SA?
            # If we just do hill climbing, we might get stuck.
            # Let's add a small probability to accept worse moves to escape local minima.
            # But we need to track T.
            # Let's keep it simple: Hill Climbing with restarts.
            
            # Check for stall
            # If no improvement in last 1000 steps, restart?
            # For now, just continue.
            
    return best_s
