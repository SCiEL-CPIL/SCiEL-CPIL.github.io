import numpy as np
import time
from scipy.optimize import minimize

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
    
    # Simulated annealing parameters
    np.random.seed(42)
    T_start = 0.01
    T_end = 1e-6
    max_iterations = 300000
    no_improve_count = 0
    patience = 5000
    
    # Current state (may differ from best)
    current_step = best_step.copy()
    current_c3 = best_c3
    
    for iteration in range(max_iterations):
        # Check deadline with safety margin
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Geometric cooling
        if max_iterations > 1:
            alpha = iteration / max_iterations
            T = T_start * (T_end / T_start) ** alpha
        else:
            T = T_start
        
        # Choose number of flips: more aggressive at higher temperatures
        r = np.random.random()
        if r < 0.6:
            n_flips = 1
        elif r < 0.85:
            n_flips = 2
        else:
            n_flips = 3
        
        # Select random indices to flip (without replacement)
        idxs = np.random.choice(N, size=n_flips, replace=False)
        old_vals = current_step[idxs].copy()
        current_step[idxs] = -current_step[idxs]
        
        new_c3 = _compute_c3(current_step)
        
        # Accept or reject based on simulated annealing criterion
        if new_c3 < current_c3:
            # Always accept improvements
            current_c3 = new_c3
            no_improve_count = 0
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Accept uphill moves with probability exp(-delta/T)
            delta = new_c3 - current_c3
            if T > 1e-12:
                accept_prob = np.exp(-delta / T)
            else:
                accept_prob = 0.0
            
            if np.random.random() < accept_prob:
                current_c3 = new_c3
                no_improve_count = 0
                if new_c3 < best_c3:
                    best_c3 = new_c3
                    best_step = current_step.copy()
            else:
                # Reject: revert
                current_step[idxs] = old_vals
                no_improve_count += 1
        
        # If stuck for too long, do a larger perturbation to escape
        if no_improve_count >= patience:
            n_perturb = np.random.randint(10, 30)
            idxs = np.random.choice(N, size=n_perturb, replace=False)
            current_step[idxs] = -current_step[idxs]
            perturbed_c3 = _compute_c3(current_step)
            
            # Accept perturbation if it improves or with some probability
            if perturbed_c3 < current_c3:
                current_c3 = perturbed_c3
                if perturbed_c3 < best_c3:
                    best_c3 = perturbed_c3
                    best_step = current_step.copy()
            else:
                # Revert perturbation
                current_step[idxs] = -current_step[idxs]
            
            no_improve_count = 0
    
    return best_step
