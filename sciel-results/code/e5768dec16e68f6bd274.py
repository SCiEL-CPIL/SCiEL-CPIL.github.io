import numpy as np
import time

N = 400
W = 0.5 / N  # width of each interval

def _autoconv_max_abs(s):
    """Compute max |autoconvolution| for sign vector s (each ±1)."""
    # The function is s[i] on interval i, each of width W
    # Autoconvolution (f*f)(t) = ∫ f(τ) f(τ-t) dτ
    # Discretize: x[i] = s[i] * W represents the area of each step
    # Autoconvolution via FFT with zero-padding
    x = s.astype(np.float64) * W
    nfft = 2 * N
    X = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(X * np.conj(X), n=nfft)
    return float(np.max(np.abs(ac)))

def _make_sine_signs(freq, phase):
    """Create sign vector from sine wave zero crossings."""
    t = np.arange(N, dtype=np.float64)
    vals = np.sin(2.0 * np.pi * freq * t / N + phase)
    return np.where(vals >= 0, 1.0, -1.0)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin
    
    # Phase 1: Exhaustive search over sine-based patterns
    best_s = None
    best_val = np.inf
    
    for freq in range(1, 25):
        for phase_deg in range(0, 360, 10):
            if time.time() > deadline:
                break
            phase = np.deg2rad(phase_deg)
            s = _make_sine_signs(freq, phase)
            val = _autoconv_max_abs(s)
            if val < best_val:
                best_val = val
                best_s = s.copy()
        if time.time() > deadline:
            break
    
    if best_s is None:
        best_s = np.ones(N)
        best_val = _autoconv_max_abs(best_s)
    
    # Phase 2: Simulated annealing with block flips
    s = best_s.copy()
    current_val = best_val
    best_s_anneal = s.copy()
    best_val_anneal = current_val
    
    T = 0.3
    T_min = 0.0005
    alpha = 0.9995
    
    while time.time() < deadline:
        # Pick a block size from a set
        block_sizes = [4, 8, 16, 32, 64]
        bs = block_sizes[np.random.randint(0, len(block_sizes))]
        
        if bs >= N:
            bs = N // 2
        
        start = np.random.randint(0, N - bs + 1)
        
        # Flip the block
        s[start:start+bs] *= -1.0
        
        new_val = _autoconv_max_abs(s)
        delta = new_val - current_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = new_val
            if new_val < best_val_anneal:
                best_val_anneal = new_val
                best_s_anneal = s.copy()
        else:
            # Revert
            s[start:start+bs] *= -1.0
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    if best_val_anneal < best_val:
        best_s = best_s_anneal
        best_val = best_val_anneal
    
    return best_s
