import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # Safety margin: evaluator kills at 150s

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 2 * N  # fine grid points
    dx = 0.5 / M
    
    # Pre-compute mapping indices once (cached)
    if not hasattr(_compute_c3, '_i_indices'):
        m_indices = np.arange(M)
        i_indices = (m_indices * N // M)
        i_indices = np.clip(i_indices, 0, N-1)
        _compute_c3._i_indices = i_indices
    
    i_indices = _compute_c3._i_indices
    f_fine = signed_step[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _compute_c3_soft(signed_step, beta=50.0):
    """
    Compute a soft-max approximation of C3 using log-sum-exp.
    This provides a smoother objective for gradient-guided search.
    """
    M = 2 * N
    dx = 0.5 / M
    
    if not hasattr(_compute_c3_soft, '_i_indices'):
        m_indices = np.arange(M)
        i_indices = (m_indices * N // M)
        i_indices = np.clip(i_indices, 0, N-1)
        _compute_c3_soft._i_indices = i_indices
    
    i_indices = _compute_c3_soft._i_indices
    f_fine = signed_step[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    abs_autoconv = np.abs(autoconv)
    
    # Soft max via log-sum-exp
    max_val = np.max(abs_autoconv)
    if max_val < 1e-15:
        return max_val
    
    shifted = abs_autoconv - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    return log_sum_exp

def _compute_autoconv(signed_step):
    """
    Compute the autoconvolution for a signed step function.
    """
    M = 2 * N
    dx = 0.5 / M
    
    if not hasattr(_compute_autoconv, '_i_indices'):
        m_indices = np.arange(M)
        i_indices = (m_indices * N // M)
        i_indices = np.clip(i_indices, 0, N-1)
        _compute_autoconv._i_indices = i_indices
    
    i_indices = _compute_autoconv._i_indices
    f_fine = signed_step[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return autoconv

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    np.random.seed(42)
    
    # Try several random initializations
    for trial in range(20):
        if time.time() - start_time > DEADLINE - 15.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Phase 1: Local search with multi-flip moves
    max_iterations = 50000
    no_improve_count = 0
    patience = 1500
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE - 10.0:
            break
        
        # Adaptive move size: increase flips when stuck
        stuck_ratio = min(no_improve_count / patience, 1.0)
        r = np.random.random()
        
        if r < 0.7 - 0.3 * stuck_ratio:
            num_flips = 1
        elif r < 0.9 - 0.2 * stuck_ratio:
            num_flips = 2
        else:
            num_flips = 3 + int(stuck_ratio * 2)  # Up to 5 flips when very stuck
        
        num_flips = min(num_flips, 5)
        
        if num_flips == 1:
            idxs = [np.random.randint(N)]
        else:
            idxs = list(np.random.choice(N, size=num_flips, replace=False))
        
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
            if time.time() - start_time > DEADLINE - 15.0:
                break
            
            # Perturbation: random restart in a neighborhood
            perturbation = np.random.choice([-1, 1], size=N).astype(np.int8)
            # Mix: keep some of the current best, replace others
            keep_ratio = 0.5 + 0.3 * np.random.random()
            mask = np.random.random(N) < keep_ratio
            candidate = best_step.copy()
            candidate[~mask] = perturbation[~mask]
            c3_val = _compute_c3(candidate)
            if c3_val < best_c3:
                best_step = candidate
                best_c3 = c3_val
            no_improve_count = 0
    
    # Phase 2: Gradient-informed polish using soft surrogate
    if time.time() - start_time < DEADLINE - 20.0:
        # Use soft C3 to guide flips toward reducing the max
        beta = 30.0
        soft_c3 = _compute_c3_soft(best_step, beta)
        
        polish_iterations = 5000
        no_improve_soft = 0
        soft_patience = 500
        
        for iteration in range(polish_iterations):
            if time.time() - start_time > DEADLINE - 10.0:
                break
            
            # Try single flips and evaluate using soft C3 for speed
            idx = np.random.randint(N)
            old_val = best_step[idx]
            best_step[idx] = -best_step[idx]
            
            new_soft = _compute_c3_soft(best_step, beta)
            
            if new_soft < soft_c3:
                soft_c3 = new_soft
                # Also check exact C3
                new_c3_exact = _compute_c3(best_step)
                if new_c3_exact < best_c3:
                    best_c3 = new_c3_exact
                no_improve_soft = 0
            else:
                best_step[idx] = old_val
                no_improve_soft += 1
            
            if no_improve_soft >= soft_patience:
                # Reset patience with a small perturbation
                if time.time() - start_time > DEADLINE - 15.0:
                    break
                idxs = np.random.choice(N, size=3, replace=False)
                for idx in idxs:
                    best_step[idx] = -best_step[idx]
                new_c3 = _compute_c3(best_step)
                if new_c3 < best_c3:
                    best_c3 = new_c3
                no_improve_soft = 0
    
    return best_step
