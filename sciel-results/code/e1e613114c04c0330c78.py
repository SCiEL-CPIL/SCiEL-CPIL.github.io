"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_max_abs(s):
    """Compute max |autocorrelation| over non-zero lags for a length-N ±1 sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k] for k=0..N-1
    if N <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))

def _pairwise_local_search(s, max_time=5.0):
    """Local search using pairwise flips to escape single-flip local minima."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        # Try flipping pairs of elements
        for i in range(N - 1):
            for j in range(i + 1, min(i + 50, N)):  # Only check nearby pairs for efficiency
                s_old_i, s_old_j = s[i], s[j]
                s[i], s[j] = -s_old_i, -s_old_j
                new_val = _autocorr_max_abs(s)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                else:
                    s[i], s[j] = s_old_i, s_old_j
        if not improved:
            break
    return s, current_val

def _simulated_annealing_pairwise(s, max_time=10.0):
    """Simulated annealing with pairwise flips for better exploration."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    best_s = s.copy()
    best_val = current_val
    start_time = time.time()
    T = 2.0
    cooling = 0.998
    
    while time.time() - start_time < max_time and T > 0.05:
        # Pick two random positions to flip
        i, j = np.random.choice(N, size=2, replace=False)
        s_old_i, s_old_j = s[i], s[j]
        s[i], s[j] = -s_old_i, -s_old_j
        new_val = _autocorr_max_abs(s)
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = s.copy()
        else:
            s[i], s[j] = s_old_i, s_old_j  # revert
        T *= cooling
    
    return best_s, best_val

def _multi_bit_sa(s, max_time=10.0, num_bits=3):
    """Simulated annealing flipping multiple bits at once for larger moves."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    best_s = s.copy()
    best_val = current_val
    start_time = time.time()
    T = 3.0
    cooling = 0.997
    
    while time.time() - start_time < max_time and T > 0.1:
        # Pick multiple random positions to flip
        indices = np.random.choice(N, size=num_bits, replace=False)
        old_vals = s[indices].copy()
        s[indices] = -old_vals
        new_val = _autocorr_max_abs(s)
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = s.copy()
        else:
            s[indices] = old_vals  # revert
        T *= cooling
    
    return best_s, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Generate diverse initial structures
    candidates = []
    
    # 1. Pseudo-random ±1 sequence
    np.random.seed(42)
    s_rand = np.random.choice([-1, 1], size=N)
    candidates.append(s_rand)
    
    # 2. Alternating pattern
    s_alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
    candidates.append(s_alt)
    
    # 3. Block pattern
    s_block = np.tile([1]*10 + [-1]*10, N // 20)
    if len(s_block) < N:
        s_block = np.pad(s_block, (0, N - len(s_block)), mode='wrap')
    candidates.append(s_block)
    
    # 4. Sine-based pattern
    t = np.linspace(0, 2 * np.pi * 10, N)
    s_sine = np.sign(np.sin(t))
    s_sine[s_sine == 0] = 1
    candidates.append(s_sine)
    
    # 5. Another random seed
    np.random.seed(123)
    s_rand2 = np.random.choice([-1, 1], size=N)
    candidates.append(s_rand2)
    
    # 6. Mersenne-like pattern based on known good sequences
    np.random.seed(999)
    s_mersenne = np.random.choice([-1, 1], size=N)
    candidates.append(s_mersenne)
    
    best_s = None
    best_val = float('inf')
    
    for s_init in candidates:
        if time.time() > deadline:
            break
        
        # Pairwise local search first
        s_ls, val_ls = _pairwise_local_search(s_init, max_time=4.0)
        if val_ls < best_val:
            best_val = val_ls
            best_s = s_ls.copy()
        
        # Simulated annealing with pairwise flips if time permits
        if time.time() < deadline - 20:
            s_sa, val_sa = _simulated_annealing_pairwise(s_init, max_time=8.0)
            if val_sa < best_val:
                best_val = val_sa
                best_s = s_sa.copy()
        
        # Multi-bit SA for larger moves
        if time.time() < deadline - 10:
            s_mb, val_mb = _multi_bit_sa(best_s, max_time=5.0, num_bits=3)
            if val_mb < best_val:
                best_val = val_mb
                best_s = s_mb.copy()
        
        # Final pairwise local search on best
        if time.time() < deadline - 5:
            s_final, val_final = _pairwise_local_search(best_s, max_time=3.0)
            if val_final < best_val:
                best_val = val_final
                best_s = s_final.copy()
    
    if best_s is None:
        best_s = np.ones(N)
    
    # Final refinement
    if time.time() < deadline:
        s_final, val_final = _pairwise_local_search(best_s, max_time=2.0)
        if val_final < best_val:
            best_s = s_final
    
    return best_s
# EVOLVE-BLOCK-END
