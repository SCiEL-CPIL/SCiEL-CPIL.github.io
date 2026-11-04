"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _thue_morse(n):
    """
    Generate the Thue-Morse sequence of length n, mapped to +1/-1.
    The Thue-Morse sequence t_n is defined as the parity of the number of 1s
    in the binary representation of n.
    t: 0, 1, 1, 0, 1, 0, 0, 1, ...
    Mapped: +1, -1, -1, +1, ...
    """
    # Start with [0]
    seq = np.array([0], dtype=int)
    # Double the sequence until it is at least length n
    # t_{2k} = t_k, t_{2k+1} = 1 - t_k
    # This is equivalent to seq = concat(seq, 1-seq)
    while len(seq) < n:
        seq = np.concatenate([seq, 1 - seq])
    # Take first n elements
    seq = seq[:n]
    # Map 0 -> 1, 1 -> -1
    # (2 * (1 - seq) - 1) ? 
    # If seq=0, want 1. 2*1-1=1.
    # If seq=1, want -1. 2*0-1=-1.
    return (2 * (1 - seq) - 1).astype(float)

def _compute_score(f):
    """
    Compute the maximum absolute value of the autoconvolution 
    (excluding the zero-lag term).
    """
    n = len(f)
    if n == 0:
        return 0.0
    
    # Linear convolution via FFT
    # Size needs to be at least 2n - 1. 2n is a power of 2 often, or close.
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    
    F = np.fft.fft(f_pad, size)
    conv = np.fft.ifft(F * F).real
    
    # conv[0] is the zero-lag term (sum of squares)
    # We want max |conv[k]| for k != 0
    # Due to symmetry, we only need to check up to n
    # conv[k] for k in 1..n-1 and conv[n] (which is 0 for linear conv of length n)
    # Actually, linear conv of length n with itself has length 2n-1.
    # Indices 0 to 2n-2.
    # We exclude index 0.
    # Max abs of conv[1:]
    return np.max(np.abs(conv[1:]))

def _local_search(f, deadline):
    """
    Greedy local search: flip bits if it reduces the score.
    """
    current_f = f.copy()
    current_score = _compute_score(current_f)
    
    # Track number of passes to avoid infinite loops if stuck in oscillation
    # (Greedy on max-abs is usually monotonic decreasing, so no oscillation expected)
    
    while time.time() < deadline:
        improved = False
        # Iterate through all indices
        for i in range(N):
            # Flip the bit
            old_val = current_f[i]
            current_f[i] = -old_val
            
            # Calculate new score
            new_score = _compute_score(current_f)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert the flip
                current_f[i] = old_val
        
        if not improved:
            break
            
    return current_f, current_score

def construct():
    start_time = time.time()
    # Keep a safety margin of 10 seconds before the 150s limit
    deadline = start_time + 140.0
    
    best_f = np.ones(N)
    best_score = _compute_score(best_f)
    
    # 1. Thue-Morse Initialization
    # Thue-Morse sequences have very low autocorrelation peaks.
    tm_seq = _thue_morse(N)
    f_tm = tm_seq.copy()
    f_tm, score_tm = _local_search(f_tm, deadline)
    if score_tm < best_score:
        best_score = score_tm
        best_f = f_tm.copy()
        
    # 2. Windows of a longer Thue-Morse sequence
    # Truncating TM at different points might yield different local minima.
    # Generate a longer TM sequence
    tm_long = _thue_morse(512)
    
    # Try different starting windows
    # Step size 10 to sample different phases
    for start_idx in range(0, 512 - N + 1, 10):
        if time.time() > deadline - 5:
            break
        f_window = tm_long[start_idx:start_idx+N].copy()
        f_window, score_window = _local_search(f_window, deadline)
        if score_window < best_score:
            best_score = score_window
            best_f = f_window.copy()
            
    # 3. Random Restart Local Search
    # Use random seeds to escape potential local minima of TM-based starts.
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        f_rand = np.random.choice([-1, 1], size=N)
        f_rand, score_rand = _local_search(f_rand, deadline)
        if score_rand < best_score:
            best_score = score_rand
            best_f = f_rand.copy()
            
    return best_f

# EVOLVE-BLOCK-END
