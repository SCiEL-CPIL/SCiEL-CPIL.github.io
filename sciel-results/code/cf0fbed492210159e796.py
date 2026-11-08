import numpy as np
import time

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_c3_exact(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 2 * N  # fine grid points
    dx = 0.5 / M
    
    # Vectorized construction of fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices].astype(np.float64)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _compute_c3_soft(signed_step, beta=50.0):
    """
    Compute a smooth surrogate for C3 using log-sum-exp.
    This provides a differentiable approximation to the max.
    """
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices].astype(np.float64)
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    abs_autoconv = np.abs(autoconv)
    
    # Log-sum-exp for smooth max approximation
    # soft_max = (1/beta) * log(sum(exp(beta * x)))
    # For numerical stability, subtract the max
    max_val = np.max(abs_autoconv)
    if max_val > 0:
        shifted = beta * (abs_autoconv - max_val)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_max = max_val + log_sum / beta
    else:
        soft_max = 0.0
    
    return soft_max

def _compute_autoconv(signed_step):
    """Compute the autoconvolution array for reuse."""
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices].astype(np.float64)
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return autoconv

def _c3_from_autoconv(autoconv):
    """Get C3 from precomputed autoconvolution."""
    return np.max(np.abs(autoconv))

def _soft_c3_from_autoconv(autoconv, beta=50.0):
    """Get soft C3 from precomputed autoconvolution."""
    abs_autoconv = np.abs(autoconv)
    max_val = np.max(abs_autoconv)
    if max_val > 0:
        shifted = beta * (abs_autoconv - max_val)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_max = max_val + log_sum / beta
    else:
        soft_max = 0.0
    return soft_max

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3_exact = _compute_c3_exact(best_step)
    
    np.random.seed(42)
    
    # Try random initializations
    for trial in range(10):
        if time.time() - start_time > DEADLINE - 15.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3_exact(random_step)
        if c3_val < best_c3_exact:
            best_step = random_step.copy()
            best_c3_exact = c3_val
    
    # Phase 1: Local search using soft surrogate for guidance
    # The soft surrogate provides smoother landscape for exploration
    max_iterations_phase1 = 80000
    no_improve_count = 0
    patience = 1500
    beta = 50.0  # Temperature for soft-max
    
    # Precompute autoconv for current best
    best_autoconv = _compute_autoconv(best_step)
    best_soft_c3 = _soft_c3_from_autoconv(best_autoconv, beta)
    
    for iteration in range(max_iterations_phase1):
        if time.time() - start_time > DEADLINE - 20.0:
            break
        
        r = np.random.random()
        if r < 0.75:
            num_flips = 1
        elif r < 0.92:
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
        
        # Compute new autoconv and soft C3
        new_autoconv = _compute_autoconv(best_step)
        new_soft_c3 = _soft_c3_from_autoconv(new_autoconv, beta)
        
        if new_soft_c3 < best_soft_c3:
            # Accept based on soft surrogate, but track exact C3
            best_soft_c3 = new_soft_c3
            best_autoconv = new_autoconv
            new_exact_c3 = _c3_from_autoconv(new_autoconv)
            if new_exact_c3 < best_c3_exact:
                best_c3_exact = new_exact_c3
            no_improve_count = 0
        else:
            # Revert
            for i, idx in enumerate(idxs):
                best_step[idx] = old_vals[i]
            no_improve_count += 1
        
        if no_improve_count >= patience:
            if time.time() - start_time > DEADLINE - 15.0:
                break
            # Perturbation: flip a random subset
            perturbation = np.random.choice([-1, 1], size=N).astype(np.int8)
            mask = np.random.random(N) < 0.25
            candidate = best_step.copy()
            candidate[mask] = perturbation[mask]
            cand_autoconv = _compute_autoconv(candidate)
            cand_soft_c3 = _soft_c3_from_autoconv(cand_autoconv, beta)
            if cand_soft_c3 < best_soft_c3:
                best_step = candidate
                best_soft_c3 = cand_soft_c3
                best_autoconv = cand_autoconv
                cand_exact_c3 = _c3_from_autoconv(cand_autoconv)
                if cand_exact_c3 < best_c3_exact:
                    best_c3_exact = cand_exact_c3
            no_improve_count = 0
    
    # Phase 2: Polish with exact C3 objective
    # Use the best found so far and do a tighter local search
    best_autoconv = _compute_autoconv(best_step)
    best_c3_exact = _c3_from_autoconv(best_autoconv)
    
    max_iterations_phase2 = 50000
    no_improve_count = 0
    patience = 2000
    
    for iteration in range(max_iterations_phase2):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Single flip moves for precision
        idx = np.random.randint(N)
        old_val = best_step[idx]
        best_step[idx] = -best_step[idx]
        
        new_autoconv = _compute_autoconv(best_step)
        new_c3 = _c3_from_autoconv(new_autoconv)
        
        if new_c3 < best_c3_exact:
            best_c3_exact = new_c3
            best_autoconv = new_autoconv
            no_improve_count = 0
        else:
            best_step[idx] = old_val
            no_improve_count += 1
        
        if no_improve_count >= patience:
            if time.time() - start_time > DEADLINE - 5.0:
                break
            # Small perturbation
            num_perturb = np.random.randint(1, 5)
            perturb_idxs = np.random.choice(N, size=num_perturb, replace=False)
            for idx in perturb_idxs:
                best_step[idx] = -best_step[idx]
            new_autoconv = _compute_autoconv(best_step)
            new_c3 = _c3_from_autoconv(new_autoconv)
            if new_c3 < best_c3_exact:
                best_c3_exact = new_c3
                best_autoconv = new_autoconv
            else:
                for idx in perturb_idxs:
                    best_step[idx] = -best_step[idx]  # Revert
            no_improve_count = 0
    
    # Final check: make sure we return a valid step function
    return best_step
