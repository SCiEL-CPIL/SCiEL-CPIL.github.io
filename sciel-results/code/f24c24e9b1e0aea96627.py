import numpy as np
import time

N = 400
LOG_SUM_EXP_BETA = 50.0

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] using FFT."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _smooth_max_abs_acf(A, beta=LOG_SUM_EXP_BETA):
    """Smooth surrogate for max|A[1:]} using log-sum-exp."""
    x = np.abs(A[1:])
    m = np.max(x)
    if m == 0:
        return 0.0
    # log-sum-exp for stability
    shifted = x - m
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta + m
    return float(log_sum)

def _max_abs_acf_from_array(A):
    """Return the max absolute value of A[1:]."""
    return float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # Pre-compute FFT size
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    def _eval(s):
        s_f = s.astype(np.float64)
        F = np.fft.rfft(s_f, n=n_fft)
        P = F * np.conj(F)
        acf = np.fft.irfft(P, n=n_fft)
        A = np.real(acf[:N])
        val = float(np.max(np.abs(A[1:])))
        return A, val

    def _eval_smooth(s):
        """Evaluate smooth surrogate objective."""
        A = _calc_acf(s)
        val_smooth = _smooth_max_abs_acf(A)
        val_exact = float(np.max(np.abs(A[1:])))
        return A, val_smooth, val_exact

    def _metropolis_hastings(s, A, val, t_max=5.0):
        """Metropolis-Hastings with smooth surrogate for diverse exploration."""
        T = 20.0
        T_min = 0.01
        s = s.copy()
        A = A.copy()
        val = val
        
        # Smooth objective value for MH acceptance
        _, val_smooth, _ = _eval_smooth(s)
        
        # Block sizes to try - favor smaller blocks for finer control
        block_sizes = [1, 1, 1, 2, 2, 4, 8, 16]
        
        while time.time() - start_time < deadline and T > T_min:
            # Choose block size
            block_size = block_sizes[np.random.randint(0, len(block_sizes))]
            
            # Choose random start position for block
            start_idx = np.random.randint(0, N - block_size + 1)
            
            # Flip the block
            old_block = s[start_idx:start_idx + block_size].copy()
            s[start_idx:start_idx + block_size] = -old_block
            
            A_new, val_smooth_new, val_exact_new = _eval_smooth(s)
            
            delta_smooth = val_smooth_new - val_smooth
            
            if delta_smooth < 0:
                A = A_new
                val = val_exact_new
                val_smooth = val_smooth_new
            else:
                p = np.exp(-delta_smooth / T)
                if np.random.random() < p:
                    A = A_new
                    val = val_exact_new
                    val_smooth = val_smooth_new
                else:
                    s[start_idx:start_idx + block_size] = old_block
            
            # Cool down slowly for more exploration
            T *= 0.999
            
        return s, A, val

    def _greedy_polish(s, A, val, t_max=2.0):
        """Greedy local search: only accept improvements."""
        s = s.copy()
        A = A.copy()
        val = val
        
        # Try single-bit flips first, then small blocks
        move_types = [1, 2, 4, 8]
        
        while time.time() - start_time < deadline:
            improved = False
            for move_size in move_types:
                if move_size == 1:
                    # Try all single-bit flips
                    for i in range(N):
                        if time.time() - start_time > deadline:
                            return s, A, val
                        s[i] = -s[i]
                        A_new, val_new = _eval(s)
                        if val_new < val:
                            A = A_new
                            val = val_new
                            improved = True
                        else:
                            s[i] = -s[i]  # revert
                else:
                    # Try random block flips of this size
                    for _ in range(50):
                        if time.time() - start_time > deadline:
                            return s, A, val
                        start_idx = np.random.randint(0, N - move_size + 1)
                        old_block = s[start_idx:start_idx + move_size].copy()
                        s[start_idx:start_idx + move_size] = -old_block
                        A_new, val_new = _eval(s)
                        if val_new < val:
                            A = A_new
                            val = val_new
                            improved = True
                        else:
                            s[start_idx:start_idx + move_size] = old_block
            
            if not improved:
                break
        
        return s, A, val

    # Main loop: multiple restarts with MH + greedy polish
    num_restarts = 0
    while time.time() < deadline - 10.0:
        # Random initialization
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A, val = _eval(s)
        
        # Run Metropolis-Hastings with smooth surrogate
        remaining = deadline - time.time()
        mh_time = min(remaining * 0.7, 20.0)
        s, A, val = _metropolis_hastings(s, A, val, t_max=mh_time)
        
        # Greedy polish
        s, A, val = _greedy_polish(s, A, val)
        
        if val < best_val:
            best_val = val
            best_seq = s.copy()
        
        num_restarts += 1
        
        # If we're getting close to deadline, do one more quick polish
        if time.time() > deadline - 5.0:
            break
    
    # Final polish on the best sequence if time permits
    if time.time() < deadline - 2.0:
        best_A = _calc_acf(best_seq)
        best_seq, _, best_val = _greedy_polish(best_seq, best_A, best_val, t_max=2.0)
    
    return best_seq
