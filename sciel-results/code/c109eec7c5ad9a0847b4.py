import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    """
    M = 4 * N
    dx = (0.5) / M
    
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    current_step = np.ones(N, dtype=np.int8)
    current_c3 = _compute_c3(current_step)
    
    # Track best solution
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # Greedy Hill Climbing (to avoid degrading the optimal All-Ones solution)
    # If the metric is noisy, we might need to be very careful.
    # But since All-Ones is the theoretical min, we expect no moves.
    
    np.random.seed(42)
    
    # Try a few moves to see if any improve (unlikely)
    for _ in range(1000):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        idx = np.random.randint(N)
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3(current_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            best_step = current_step.copy()
            current_c3 = new_c3
        else:
            # Revert
            current_step[idx] = old_val
            # If we didn't improve, and we are at the global min, 
            # we can stop early if we detect we are stuck at the min.
            # But for now, just continue.

    return best_step
