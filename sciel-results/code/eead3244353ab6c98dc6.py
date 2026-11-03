"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin
    
    def fft_autoconv(f):
        """Compute autoconvolution via FFT. Returns array of length 2N-1."""
        n = len(f)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        F = np.fft.rfft(f, size)
        conv = np.fft.irfft(F * np.conj(F), size)[:2*n-1]
        return conv
    
    def max_abs_conv(f):
        """Compute max absolute autoconvolution value using FFT."""
        conv = fft_autoconv(f)
        return np.max(np.abs(conv))
    
    def score(f):
        """Compute the ratio: max_abs_autoconv / (sum(f))^2"""
        S = np.sum(f)
        if abs(S) < 1e-10:
            return 1e10  # penalty for zero sum
        return max_abs_conv(f) / (S * S)
    
    # Start with all +1s
    f = np.ones(N, dtype=np.float64)
    best_f = f.copy()
    best_score = score(f)
    
    np.random.seed(42)
    
    # Phase 1: Try structured initializations
    inits = []
    inits.append(np.ones(N, dtype=np.float64))
    
    # Random sign patterns
    for _ in range(10):
        inits.append(np.random.choice([-1.0, 1.0], size=N).astype(np.float64))
    
    # Block-structured patterns (alternating blocks)
    for block_size in [2, 4, 8, 16, 32, 64]:
        if time.time() > deadline:
            break
        pattern = np.ones(N)
        for i in range(0, N, 2 * block_size):
            pattern[i:i+block_size] = -1.0
        inits.append(pattern)
    
    for init_f in inits:
        if time.time() > deadline:
            break
        
        cur_score = score(init_f)
        if cur_score < best_score:
            best_f = init_f.copy()
            best_score = cur_score
    
    # Phase 2: Local search with FFT-based scoring on best candidate
    f = best_f.copy()
    cur_score = best_score
    
    # Greedy coordinate descent: try flipping each position, keep if improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = f[i]
            f[i] = -old_val
            new_score = score(f)
            if new_score < cur_score - 1e-12:
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_f = f.copy()
                improved = True
            else:
                f[i] = old_val
    
    # Phase 3: Random restart local search with limited iterations
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Do a few more random restarts with shorter local search
        n_restarts = int(min(20, remaining_time / 3.0))
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            f = np.random.choice([-1.0, 1.0], size=N).astype(np.float64)
            cur_score = score(f)
            
            # Short greedy local search
            for _ in range(3):  # 3 passes
                if time.time() > deadline:
                    break
                for i in range(0, N, 2):  # random subset
                    if time.time() > deadline:
                        break
                    idx = np.random.randint(N)
                    old_val = f[idx]
                    f[idx] = -old_val
                    new_score = score(f)
                    if new_score < cur_score - 1e-12:
                        cur_score = new_score
                    else:
                        f[idx] = old_val
            
            if cur_score < best_score:
                best_score = cur_score
                best_f = f.copy()
    
    return best_f

# EVOLVE-BLOCK-END
