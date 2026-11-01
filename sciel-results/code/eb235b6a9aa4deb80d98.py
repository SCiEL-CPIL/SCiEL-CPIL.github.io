"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h.astype(np.float64), n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    return np.max(np.abs(acf[1:]))


def _local_search_discrete(h, deadline, rng):
    """
    Discrete local search: try flipping each bit, keep if it improves.
    Uses a deadline to stop early.
    """
    n = len(h)
    best_val = _max_sidelobe(h)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_i = -1
        best_new_val = best_val
        
        # Randomize order to avoid bias
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            # Try flipping bit i
            h[i] = -h[i]
            val = _max_sidelobe(h)
            h[i] = -h[i]  # revert
            
            if val < best_new_val:
                best_new_val = val
                best_i = i
        
        if best_i >= 0 and best_new_val < best_val:
            h[best_i] = -h[best_i]
            best_val = best_new_val
            improved = True
    
    return h, best_val


def _simulated_annealing(h, deadline, rng):
    """
    Simulated Annealing for discrete optimization.
    """
    n = len(h)
    current_val = _max_sidelobe(h)
    best_val = current_val
    best_h = h.copy()
    
    T_start = 10.0
    T_end = 0.01
    steps = 50000
    cooling = (T_end / T_start) ** (1.0 / steps)
    T = T_start
    
    no_improve = 0
    stagnation_limit = 5000
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        # Pick random bit to flip
        i = rng.randint(n)
        h[i] = -h[i]
        
        new_val = _max_sidelobe(h)
        delta = new_val - current_val
        
        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            h[i] = -h[i]  # revert
            
        T *= cooling
        
        # Early stop if stuck for too long
        if no_improve > stagnation_limit:
            break
            
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_val = np.inf
    
    # Phase 1: Generate diverse starting points
    # Strategy: Mix of random, structured (Walsh-like), and all-ones
    candidates = []
    
    # 1. Random
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 2. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 3. Alternating
    candidates.append(np.array([1.0, -1.0] * (N // 2)))
    
    # 4. Block patterns
    for block_size in [4, 8, 16]:
        pattern = np.ones(N // block_size)
        pattern[1::2] = -1.0
        h_block = np.repeat(pattern, block_size)[:N]
        candidates.append(h_block)
        
    # 5. Walsh-Hadamard like (first few rows)
    # Simple construction: bit-reversal based
    def walsh_row(k, n):
        h = np.zeros(n)
        for i in range(n):
            h[i] = 1.0 if ((i & k) == 0) else -1.0
        return h
    
    for k in [1, 2, 4, 8, 16]:
        candidates.append(walsh_row(k, N))
        
    # 6. More randoms
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Phase 2: Local search on each candidate
    for i, h in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        h_copy = h.copy()
        h_ls, val_ls = _local_search_discrete(h_copy, deadline - 20, rng)
        
        if val_ls < best_val:
            best_val = val_ls
            best_h = h_ls.copy()
            
        # Phase 3: Simulated Annealing on the best so far
        if time.time() < deadline - 5:
            h_sa = best_h.copy()
            h_sa_res, val_sa = _simulated_annealing(h_sa, deadline - 5, rng)
            
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa_res.copy()
                
            # Final local search on SA result
            if time.time() < deadline - 2:
                h_final = best_h.copy()
                h_final, val_final = _local_search_discrete(h_final, deadline - 2, rng)
                if val_final < best_val:
                    best_val = val_final
                    best_h = h_final.copy()
                    
    # Fallback
    if best_h is None:
        best_h = rng.choice([-1.0, 1.0], size=N).astype(np.float64)
        
    return best_h
# EVOLVE-BLOCK-END
