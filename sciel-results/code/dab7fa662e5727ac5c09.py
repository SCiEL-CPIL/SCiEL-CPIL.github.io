"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _compute_c3(signs):
    """Compute the C3 upper bound for a signed step function.
    Uses FFT to compute linear autoconvolution, returns sum of squares.
    """
    n = len(signs)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    
    f = np.zeros(fft_size)
    f[:n] = signs
    F = np.fft.fft(f)
    power = F * np.conj(F)
    autoconv = np.fft.ifft(power).real
    
    # Linear autoconvolution lags
    pos = autoconv[:n]
    neg = autoconv[fft_size-n+1:fft_size]
    all_lags = np.concatenate([neg, pos])
    
    return np.sum(all_lags ** 2)


def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    np.random.seed(42)
    best_signs = np.random.choice([-1, 1], size=N)
    best_c3 = _compute_c3(best_signs)
    
    current_signs = best_signs.copy()
    current_c3 = best_c3
    
    T_start = 1.0
    T_end = 0.005
    n_iterations = 300000
    
    it = 0
    while it < n_iterations:
        if time.time() > deadline:
            break
        
        # Propose move
        use_block = np.random.random() < 0.25
        if use_block:
            block_len = np.random.randint(1, 30)
            start = np.random.randint(0, N - block_len + 1)
            old_vals = current_signs[start:start+block_len].copy()
            current_signs[start:start+block_len] *= -1
        else:
            idx = np.random.randint(0, N)
            old_vals = np.array([current_signs[idx]])
            current_signs[idx] *= -1
        
        new_c3 = _compute_c3(current_signs)
        delta = new_c3 - current_c3
        
        T = T_start * (T_end / T_start) ** (it / n_iterations)
        
        accept = delta < 0 or (T > 1e-10 and np.random.random() < np.exp(-delta / T))
        
        if accept:
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_signs = current_signs.copy()
        else:
            # Revert
            if use_block:
                current_signs[start:start+block_len] = old_vals
            else:
                current_signs[idx] = old_vals[0]
        
        it += 1
    
    # Local polish
    if time.time() < deadline - 3:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            for idx in range(N):
                if time.time() > deadline - 3:
                    break
                current_signs[idx] *= -1
                c3 = _compute_c3(current_signs)
                if c3 < best_c3:
                    best_c3 = c3
                    best_signs = current_signs.copy()
                else:
                    current_signs[idx] *= -1
    
    return best_signs
# EVOLVE-BLOCK-END
