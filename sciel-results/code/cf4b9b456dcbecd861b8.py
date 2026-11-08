import numpy as np
import time
import math

N = 400
DEADLINE = 145.0
M = 2 * N  # Fine grid resolution (2x N is sufficient for step function)
dx = 0.5 / M

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    # Vectorized construction of fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    f_fine = signed_step[i_indices].astype(np.float64)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    np.random.seed(42)
    
    # Try a few random initializations to potentially find a better starting point
    for trial in range(5):
        if time.time() - start_time > DEADLINE - 10.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Local search with single and multi-flip moves
    max_iterations = 50000
    no_improve_count = 0
    patience = 3000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        r = np.random.random()
        if r < 0.7:
            num_flips = 1
        elif r < 0.9:
            num_flips = 2
        else:
            num_flips = 3
        
        if num_flips == 1:
            idxs = [np.random.randint(N)]
        elif num_flips == 2:
            idxs = list(np.random.choice(N, size=2, replace=False))
        else:
            idxs = list(np.random.choice(N, size=3, replace=False))
        
        old_vals = best_step[idxs].copy()
        
        for idx in idxs:
            best_step[idx] = -best_step[idx]
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve_count = 0
        else:
            for i, idx in enumerate(idxs):
                best_step[idx] = old_vals[i]
            no_improve_count += 1
        
        if no_improve_count >= patience:
            if time.time() - start_time > DEADLINE - 10.0:
                break
            # Perturbation to escape local minimum
            perturbation = np.random.choice([-1, 1], size=N).astype(np.int8)
            mask = np.random.random(N) < 0.2
            candidate = best_step.copy()
            candidate[mask] = perturbation[mask]
            c3_val = _compute_c3(candidate)
            if c3_val < best_c3:
                best_step = candidate
                best_c3 = c3_val
            no_improve_count = 0
    
    # Final greedy polish
    if time.time() - start_time < DEADLINE - 1.0:
        for idx in range(N):
            if time.time() - start_time > DEADLINE - 0.5:
                break
            old_val = best_step[idx]
            best_step[idx] = -old_val
            new_c3 = _compute_c3(best_step)
            if new_c3 < best_c3:
                best_c3 = new_c3
            else:
                best_step[idx] = old_val
    
    return best_step
