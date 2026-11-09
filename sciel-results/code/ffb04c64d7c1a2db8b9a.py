"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak_sidelobe(h):
    """Compute max |R[k]| for k != 0 using FFT-based linear autocorrelation."""
    n = len(h)
    nfft = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    R = np.fft.irfft(H * np.conj(H), n=nfft)
    # Linear autocorrelation: R[0..n-1] are positive lags, R[nfft-n+1..nfft-1] negative lags
    sidelobes_pos = R[1:n]
    sidelobes_neg = R[nfft - n + 1:nfft]
    all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
    return np.max(np.abs(all_sidelobes))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to ±1."""
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        bits[i] = 1 if cnt % 2 == 0 else -1
    return bits

def _generate_hadamard_row(n, row_idx):
    """Generate a Hadamard-like sequence using Walsh functions."""
    # Simple recursive Hadamard construction
    h = np.ones(1)
    while len(h) < n:
        h = np.concatenate([h, h])
        h = np.concatenate([h, -h])
    # Take the row_idx-th row (0-indexed) from the full Hadamard matrix
    # For simplicity, use a deterministic pattern based on bit-reversal
    result = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Use bit-reversed index for better distribution
        rev = 0
        temp = i
        for _ in range(n.bit_length()):
            rev = (rev << 1) | (temp & 1)
            temp >>= 1
        # XOR with row index
        combined = rev ^ row_idx
        # Count bits
        cnt = 0
        x = combined
        while x:
            cnt += x & 1
            x >>= 1
        result[i] = 1 if cnt % 2 == 0 else -1
    return result

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Multiple diverse initializations
    initializations = [
        _thue_morse(N).astype(np.float64),
        _generate_hadamard_row(N, 1).astype(np.float64),
        _generate_hadamard_row(N, 2).astype(np.float64),
        _generate_hadamard_row(N, 3).astype(np.float64),
        np.random.default_rng(0).choice([-1, 1], size=N).astype(np.float64),
    ]
    
    # Pick the best initialization
    best_h = None
    best_score = float('inf')
    for init in initializations:
        score = _autocorr_peak_sidelobe(init)
        if score < best_score:
            best_score = score
            best_h = init.copy()
    
    h = best_h.copy()
    
    # Hybrid simulated annealing with flips, swaps, and adjacent swaps
    T_init = 30.0
    T_final = 0.05
    n_iterations = 80000
    
    rng = np.random.default_rng(42)
    
    for it in range(n_iterations):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Annealing schedule (geometric)
        T = T_init * (T_final / T_init) ** (it / n_iterations)
        
        # Choose move type: 0=flip block, 1=swap two elements, 2=adjacent swap
        move_type = rng.integers(0, 3)
        
        h_new = h.copy()
        
        if move_type == 0:
            # Flip a block
            bs = [1, 2, 4, 8, 16, 32][rng.integers(0, 6)]
            pos = rng.integers(0, N - bs + 1)
            h_new[pos:pos+bs] *= -1
        elif move_type == 1:
            # Swap two random elements (preserves balance)
            i, j = rng.choice(N, size=2, replace=False)
            h_new[i], h_new[j] = h_new[j], h_new[i]
        else:
            # Adjacent swap
            i = rng.integers(0, N - 1)
            h_new[i], h_new[i+1] = h_new[i+1], h_new[i]
        
        # Compute new score
        score_new = _autocorr_peak_sidelobe(h_new)
        
        # Accept if better or with probability exp(-(delta)/T)
        delta = score_new - best_score
        if delta < 0 or (T > 0 and np.log(rng.random() + 1e-30) > -delta / T):
            h = h_new
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
    
    return best_h
# EVOLVE-BLOCK-END
