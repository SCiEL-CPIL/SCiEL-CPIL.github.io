import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid using vectorized repeat
    f_fine = np.repeat(signed_step, M // N)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Simulated annealing setup
    np.random.seed(42)
    
    T_initial = 0.01  # Initial temperature
    T_final = 0.0001  # Final temperature
    cooling_rate = 0.9995  # Geometric cooling
    
    current_step = best_step.copy()
    current_c3 = best_c3
    T = T_initial
    
    # Track for restarts
    no_improve_count = 0
    patience = 3000
    restart_count = 0
    max_restarts = 20
    
    while time.time() - start_time < DEADLINE - 5.0:
        # Choose number of flips
        r = np.random.random()
        if r < 0.6:
            n_flips = 1
        elif r < 0.85:
            n_flips = 2
        else:
            n_flips = 3
        
        # Select random indices to flip
        idxs = np.random.choice(N, size=n_flips, replace=False)
        old_vals = current_step[idxs].copy()
        current_step[idxs] = -current_step[idxs]
        
        new_c3 = _compute_c3(current_step)
        delta_c3 = new_c3 - current_c3
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_c3 < 0:
            # Always accept improvements
            current_c3 = new_c3
            no_improve_count = 0
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Probabilistic acceptance for uphill moves
            if T > T_final:
                prob = np.exp(-delta_c3 / T)
                if np.random.random() < prob:
                    current_c3 = new_c3
                    no_improve_count += 1
                else:
                    # Reject: revert
                    current_step[idxs] = old_vals
                    no_improve_count += 1
            else:
                # Temperature too low, just revert
                current_step[idxs] = old_vals
                no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        
        # Check for restart
        if no_improve_count >= patience:
            restart_count += 1
            if restart_count > max_restarts:
                break
            
            # Restart from best solution with higher temperature
            current_step = best_step.copy()
            current_c3 = best_c3
            T = T_initial * (0.5 ** restart_count)
            no_improve_count = 0
    
    return best_step
