"""Rewritten program using Thue-Morse initialization and Simulated Annealing."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as signed values {-1, 1}."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # t(n) = 1 if even number of 1s, -1 if odd? Or vice versa.
    # Standard TM: 0, 1, 1, 0, 1, 0, 0, 1 ...
    # Let's map 0 -> 1, 1 -> -1 or similar. The sign doesn't matter for autocorrelation magnitude.
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count bits
        # Efficient way: use bit_count or bin(i).count('1')
        # For small n, bin is fine, but let's be efficient
        seq[i] = 1 if (bin(i).count('1') % 2 == 0) else -1
    return seq.astype(np.float64)

def _compute_psl(f):
    """Compute Peak Sidelobe Level (max abs autocorrelation at non-zero lags)."""
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, n=size)
    acf = np.fft.irfft(F * np.conj(F), n=size)[:n]
    # Exclude lag 0
    if n > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def _simulated_annealing(f, time_budget=10.0, seed=42):
    """Optimize PSL using Simulated Annealing with single-bit flips."""
    f = f.copy()
    current_val = _compute_psl(f)
    best_f = f.copy()
    best_val = current_val
    
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    # Initial temperature
    T = 10.0
    cooling = 0.9995
    min_T = 0.1
    
    while time.time() - start_time < time_budget:
        if T < min_T:
            break
            
        # Randomly select an index to flip
        idx = rng.integers(0, N)
        old_sign = f[idx]
        f[idx] = -old_sign
        
        new_val = _compute_psl(f)
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_f = f.copy()
        else:
            # Revert
            f[idx] = old_sign
            
        T *= cooling
        
    return best_f, best_val

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    # Candidate initializations
    candidates = []
    
    # 1. Thue-Morse
    f_tm = _thue_morse(N)
    candidates.append(f_tm)
    
    # 2. Random initializations
    rng = np.random.default_rng(123)
    for _ in range(10):
        f_rand = rng.choice([-1.0, 1.0], size=N)
        candidates.append(f_rand)
        
    # 3. Rule 90 (from seed, just in case)
    state = np.zeros(N, dtype=np.uint8)
    state[N // 2] = 1
    for _ in range(8):
        state = np.roll(state, 1) ^ np.roll(state, -1)
    f_ca = 2 * state.astype(np.float64) - 1.0
    candidates.append(f_ca)
    
    # Evaluate initial candidates
    best_f = None
    best_val = np.inf
    for f_cand in candidates:
        val = _compute_psl(f_cand)
        if val < best_val:
            best_val = val
            best_f = f_cand.copy()
            
    # Run Simulated Annealing on the best candidate
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        best_f, best_val = _simulated_annealing(best_f, time_budget=remaining_time - 2.0, seed=42)
        
    # Try SA on the second best candidate if time permits
    # Sort candidates by initial PSL
    scored_candidates = [( _compute_psl(f), f) for f in candidates]
    scored_candidates.sort(key=lambda x: x[0])
    
    if len(scored_candidates) > 1:
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            f_second = scored_candidates[1][1]
            f_opt, val_opt = _simulated_annealing(f_second, time_budget=remaining_time - 2.0, seed=123)
            if val_opt < best_val:
                best_val = val_opt
                best_f = f_opt

    if best_f is None:
        return np.ones(N)
    return best_f
# EVOLVE-BLOCK-END
