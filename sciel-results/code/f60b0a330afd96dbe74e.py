import numpy as np
import time
import math

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 4 * N
    dx = 0.5 / M
    
    f_fine = np.repeat(signed_step, M // N)
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def construct():
    start_time = time.time()
    
    np.random.seed(42)
    
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Multi-restart simulated annealing
    num_restarts = 5
    time_per_restart = (DEADLINE - 2.0) / num_restarts
    
    for restart in range(num_restarts):
        # Check deadline
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 2.0:
            break
        
        # Initialize with random perturbation of best found so far
        current_step = best_step.copy()
        
        # Perturb with random flips
        num_flips = np.random.randint(10, 50)
        flip_indices = np.random.choice(N, size=num_flips, replace=False)
        current_step[flip_indices] *= -1
        
        current_c3 = _compute_c3(current_step)
        
        # Simulated annealing parameters for this restart
        T_initial = 0.005
        T_final = 0.00001
        cooling_rate = 0.998
        
        T = T_initial
        restart_start = time.time()
        
        while True:
            # Check deadline
            elapsed_total = time.time() - start_time
            if elapsed_total > DEADLINE - 2.0:
                break
            
            # Check if we've used up our time budget for this restart
            elapsed_restart = time.time() - restart_start
            if elapsed_restart > time_per_restart:
                break
            
            # Try flipping a random element
            idx = np.random.randint(N)
            current_step[idx] = -current_step[idx]
            
            new_c3 = _compute_c3(current_step)
            delta_c3 = new_c3 - current_c3
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta_c3 < 0:
                # Always accept improvements
                current_c3 = new_c3
                if new_c3 < best_c3:
                    best_c3 = new_c3
                    best_step = current_step.copy()
            else:
                # Probabilistic acceptance for uphill moves
                if T > 1e-12:
                    accept_prob = math.exp(-delta_c3 / T)
                    if np.random.random() < accept_prob:
                        current_c3 = new_c3
                    else:
                        # Reject: revert the flip
                        current_step[idx] = -current_step[idx]
                else:
                    # At very low temperature, only accept improvements
                    current_step[idx] = -current_step[idx]
            
            # Cool down
            T *= cooling_rate
    
    return best_step
