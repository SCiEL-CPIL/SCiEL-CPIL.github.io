"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1."""
    seq = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        # Thue-Morse: s[i] = -s[i >> 1] if (i & (i-1)) == 0 else s[i ^ (i & -i)]
        # Simpler: use the property that t(i) = parity of popcount(i)
        # t(0)=0, t(1)=1, t(2)=1, t(3)=0, ...
        # Map to ±1: if popcount(i) % 2 == 0, s=+1, else s=-1
        pass
    # Better approach: iterative construction
    seq = np.array([1], dtype=np.int8)
    while len(seq) < n:
        seq = np.concatenate([seq, -seq])
    return seq[:n]

def _compute_peak_autocorr(s):
    """Compute the maximum absolute value of autocorrelation at non-zero lags."""
    n = len(s)
    # Use FFT for efficient autocorrelation
    # Zero-pad to avoid circular effects
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(s, n=size)
    acf = np.fft.irfft(f * np.conj(f), n=size)[:n]
    # acf[0] is the sum of squares = n
    # We want max |acf[k]| for k=1..n-1
    if n > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def _local_search(s, max_iters=5000):
    """Greedy local search: flip bits that reduce peak autocorrelation."""
    s = s.copy()
    current_peak = _compute_peak_autocorr(s)
    n = len(s)
    
    # Precompute which indices to try
    indices = np.arange(n)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Shuffle order to avoid bias
        np.random.shuffle(indices)
        for i in indices:
            old_val = s[i]
            s[i] = -old_val
            new_peak = _compute_peak_autocorr(s)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                s[i] = old_val
    return s, current_peak

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with Thue-Morse
    best_s = _thue_morse(N)
    best_peak = _compute_peak_autocorr(best_s)
    
    # Try local search from Thue-Morse
    s, peak = _local_search(best_s, max_iters=2000)
    if peak < best_peak:
        best_s = s
        best_peak = peak
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Random initialization
        s_rand = np.random.choice([-1, 1], size=N).astype(np.int8)
        s_opt, peak = _local_search(s_rand, max_iters=1000)
        if peak < best_peak:
            best_s = s_opt
            best_peak = peak
        
        # Also try Thue-Morse with some random flips
        if restarts % 2 == 0:
            s_tm = _thue_morse(N).copy()
            flip_mask = np.random.random(N) < 0.1
            s_tm[flip_mask] *= -1
            s_opt, peak = _local_search(s_tm, max_iters=1000)
            if peak < best_peak:
                best_s = s_opt
                best_peak = peak
    
    return best_s
# EVOLVE-BLOCK-END
