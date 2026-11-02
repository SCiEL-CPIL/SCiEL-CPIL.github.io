"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

def _autocorrelation_max_abs(s):
    """Compute max absolute off-peak autocorrelation for a +/-1 sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    s_float = s.astype(np.float64)
    f = np.fft.fft(s_float, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:N]
    acf /= N
    
    off_peak = np.abs(acf[1:])
    if len(off_peak) == 0:
        return 0.0
    return float(np.max(off_peak))

def _score_sequence(s):
    """The score is the upper bound on C3 - we want to minimize this."""
    return _autocorrelation_max_abs(s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # Pre-compute FFT size
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    def fast_score(s):
        """Faster scoring using pre-allocated arrays."""
        s_float = s.astype(np.float64)
        f = np.fft.fft(s_float, n=n_fft)
        acf = np.fft.ifft(f * np.conj(f)).real[:N]
        acf /= N
        off_peak = np.abs(acf[1:])
        return float(np.max(off_peak))
    
    # Strategy: Use multiple restarts with different initializations
    # and aggressive local search with simulated annealing
    
    # Try structured initializations first
    initializations = []
    
    # All ones
    initializations.append(np.ones(N, dtype=np.int8))
    
    # Alternating
    initializations.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8))
    
    # Blocks of various sizes
    for block_size in [2, 3, 4, 5, 7, 8, 10]:
        init = np.ones(N, dtype=np.int8)
        for i in range(N):
            if (i // block_size) % 2 == 1:
                init[i] = -1
        initializations.append(init)
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(10):
        initializations.append(rng.choice([-1, 1], size=N).astype(np.int8))
    
    # Evaluate all initializations
    for init in initializations:
        if time.time() > deadline:
            break
        score = fast_score(init)
        if score < best_score:
            best_score = score
            best_seq = init.copy()
    
    # Local search with simulated annealing from best solution
    s = best_seq.copy()
    current_score = best_score
    
    # Simulated annealing parameters
    T_start = 0.5
    T_min = 0.001
    cooling = 0.995
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if T_start * (cooling ** it) < T_min:
            break
        
        T = T_start * (cooling ** it)
        
        # Flip one random position
        idx = rng.randint(0, N)
        old_val = s[idx]
        s[idx] = -old_val
        
        new_score = fast_score(s)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_seq = s.copy()
        else:
            delta = new_score - current_score
            if delta > 0 and rng.rand() < np.exp(-delta / T):
                current_score = new_score
            else:
                s[idx] = old_val
    
    # Second phase: more aggressive search with multi-bit flips
    if time.time() < deadline:
        s = best_seq.copy()
        current_score = best_score
        T = 0.3
        cooling = 0.997
        max_iter2 = 30000
        
        for it in range(max_iter2):
            if time.time() > deadline:
                break
            if T < T_min:
                break
            
            # Flip 2 random positions
            idxs = rng.choice(N, size=2, replace=False)
            old_vals = s[idxs].copy()
            s[idxs] = -old_vals
            
            new_score = fast_score(s)
            delta = new_score - current_score
            
            if delta < 0 or (delta > 0 and rng.rand() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_seq = s.copy()
            else:
                s[idxs] = old_vals
            
            T *= cooling
    
    # Third phase: try to refine with single flips at low temperature
    if time.time() < deadline:
        s = best_seq.copy()
        current_score = best_score
        T = 0.01
        cooling = 0.99
        max_iter3 = 20000
        
        for it in range(max_iter3):
            if time.time() > deadline:
                break
            if T < T_min:
                break
            
            idx = rng.randint(0, N)
            old_val = s[idx]
            s[idx] = -old_val
            
            new_score = fast_score(s)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_seq = s.copy()
            else:
                delta = new_score - current_score
                if delta > 0 and rng.rand() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    s[idx] = old_val
            
            T *= cooling
    
    # Return as a sequence of signed step heights
    return best_seq.tolist()
# EVOLVE-BLOCK-END
