import numpy as np
import time

N = 400

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

def _max_abs_acf_from_array(A):
    """Return the max absolute value of A[1:]."""
    return float(np.max(np.abs(A[1:])))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin: evaluator kills at 150s
    
    # Pre-compute FFT size
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    # Helper to compute A and val quickly
    def _eval(s):
        s_f = s.astype(np.float64)
        F = np.fft.rfft(s_f, n=n_fft)
        P = F * np.conj(F)
        acf = np.fft.irfft(P, n=n_fft)
        A = np.real(acf[:N])
        val = float(np.max(np.abs(A[1:])))
        return A, val

    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # Block-flip SA: flip contiguous blocks of signs to escape local minima
    def _sa_block_run(s_init, A_init, val_init, t_max, block_sizes=(4, 8, 16, 32)):
        s = s_init.copy()
        A = A_init.copy()
        val = val_init
        
        # Temperature schedule tuned for block moves
        T = 0.15
        T_min = 0.001
        cooling = 0.9998
        
        while time.time() < t_max and T > T_min:
            # Choose random block size and position
            bs = block_sizes[np.random.randint(len(block_sizes))]
            if bs >= N:
                bs = N
            i = np.random.randint(N - bs + 1)
            
            # Flip the block
            old_vals = s[i:i+bs].copy()
            s[i:i+bs] = -old_vals
            
            A_new, val_new = _eval(s)
            
            delta = val_new - val
            
            if delta < 0:
                A = A_new
                val = val_new
            else:
                # Accept with probability exp(-delta/T)
                if T > 0:
                    p = np.exp(-delta / T)
                else:
                    p = 0.0
                if np.random.random() < p:
                    A = A_new
                    val = val_new
                else:
                    s[i:i+bs] = old_vals
            
            T *= cooling
            
        return s, A, val

    # Phase 1: Random initialization with quick local search
    n_init = 8
    for i in range(n_init):
        if time.time() > deadline - 10.0:
            break
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A, val = _eval(s)
        if val < best_val:
            best_val = val
            best_seq = s.copy()
    
    # Phase 2: Block-flip SA from best found
    if time.time() < deadline - 15.0:
        s_best = best_seq.copy()
        A_best, val_best = _eval(s_best)
        s_out, A_out, val_out = _sa_block_run(s_best, A_best, val_best, deadline - 15.0)
        if val_out < best_val:
            best_val = val_out
            best_seq = s_out.copy()

    # Phase 3: Multiple random restarts with block-flip SA
    restart_count = 0
    while time.time() < deadline - 5.0:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A, val = _eval(s)
        
        # Short SA run with varied block sizes
        t_end = min(deadline - 5.0, time.time() + 3.0)
        # Use different block size sets for diversity
        if restart_count % 3 == 0:
            bs = (4, 8, 16, 32)
        elif restart_count % 3 == 1:
            bs = (8, 16, 32, 64)
        else:
            bs = (2, 4, 8, 16)
        
        s, A, val = _sa_block_run(s, A, val, t_end, block_sizes=bs)
        
        if val < best_val:
            best_val = val
            best_seq = s.copy()
        
        restart_count += 1
        
        # Safety check
        if time.time() > deadline - 3.0:
            break
    
    # Phase 4: Final polish - single-bit greedy descent from best
    if time.time() < deadline - 1.0:
        s = best_seq.copy()
        A, val = _eval(s)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                old_si = s[i]
                s[i] = -old_si
                A_new, val_new = _eval(s)
                if val_new < val:
                    A = A_new
                    val = val_new
                    improved = True
                else:
                    s[i] = old_si
            if not improved:
                break
        if val < best_val:
            best_val = val
            best_seq = s.copy()
    
    return best_seq
