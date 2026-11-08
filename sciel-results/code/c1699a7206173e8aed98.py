import numpy as np
import time

N = 400
DEADLINE = 120.0
M = 4 * N  # fine grid size
dx = 0.5 / M  # domain length is 0.5

def _compute_c3_fast(signed_step):
    """
    Fast computation of C3 = max_t |autoconv(t)| for signed step function.
    Uses vectorized fine-grid construction and FFT.
    """
    # Vectorized construction of fine grid from step function
    # Each step interval i maps to M//N fine grid points
    f_fine = np.repeat(signed_step.astype(np.float64), M // N)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return np.max(np.abs(autoconv))

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    return _compute_c3_fast(signed_step)

def construct():
    start_time = time.time()
    
    # Initialize with random signs to avoid bias towards all-ones
    np.random.seed(42)
    best_step = np.random.choice([-1, 1], size=N).astype(np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Current state for SA
    current_step = best_step.copy()
    current_c3 = best_c3
    
    # SA Parameters
    T0 = 0.5
    alpha = 0.998  # Slightly faster cooling
    T = T0
    reheat_threshold = 500
    no_improve_count = 0
    best_T_reheat = 0.5
    
    while True:
        # Check deadline with safety margin
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Reheat if stuck
        if no_improve_count > reheat_threshold:
            T = best_T_reheat
            no_improve_count = 0
            # Optional: randomize current state slightly to escape deep local minima
            if np.random.random() < 0.1:
                idxs = np.random.choice(N, size=10, replace=False)
                for i in idxs:
                    current_step[i] = -current_step[i]
                current_c3 = _compute_c3(current_step)

        # Randomly choose move type
        # Increase probability of multi-flips to escape local minima more effectively
        r = np.random.random()
        if r < 0.6:
            # Single flip
            idx = np.random.randint(N)
            old_val = current_step[idx]
            current_step[idx] = -old_val
            move_info = ('single', idx, old_val)
        elif r < 0.9:
            # Double flip
            idxs = np.random.choice(N, size=2, replace=False)
            old_vals = [current_step[i] for i in idxs]
            for i in idxs:
                current_step[i] = -current_step[i]
            move_info = ('multi', idxs, old_vals)
        else:
            # Triple flip
            idxs = np.random.choice(N, size=3, replace=False)
            old_vals = [current_step[i] for i in idxs]
            for i in idxs:
                current_step[i] = -current_step[i]
            move_info = ('multi', idxs, old_vals)
            
        new_c3 = _compute_c3(current_step)
        
        # Acceptance Criteria
        delta = new_c3 - current_c3
        if delta < 0 or (T > 1e-10 and np.random.random() < np.exp(-delta / T)):
            # Accept
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject: revert
            if move_info[0] == 'single':
                current_step[move_info[1]] = move_info[2]
            else:
                for i, v in zip(move_info[1], move_info[2]):
                    current_step[i] = v
            no_improve_count += 1
        
        # Cool down
        T *= alpha
        
        # Safety check for T underflow
        if T < 1e-10:
             T = 1e-10

    return best_step
