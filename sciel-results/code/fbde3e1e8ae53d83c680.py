"""Optimized: FFT-based autoconv + efficient delta updates for sign flips."""
import time
import numpy as np

N = 400


def _fft_autoconv(h):
    """Compute linear autoconvolution via FFT (fast)."""
    n = len(h)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    C = np.fft.irfft(H * H, n=fft_size)
    return C[:size]


def _eval(h):
    """Return (max_abs_autoconv, integral_sq, ratio)."""
    c = _fft_autoconv(h)
    max_abs = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    int_sq = s * s
    if int_sq < 1e-12:
        ratio = max_abs
    else:
        ratio = max_abs / int_sq
    return max_abs, int_sq, ratio


def _delta_flip(c, h, idx):
    """
    Compute the change in autoconvolution when flipping h[idx].
    c_new[lag] = c[lag] - 2*h[idx]*(h[idx+lag] if in range else 0) - 2*h[idx]*(h[idx-lag] if in range else 0)
    Actually, let's derive properly:
    c[lag] = sum_{j} h[j]*h[j+lag] for j such that both indices valid.
    When we flip h[idx] to -h[idx], the products involving h[idx] change:
    - Terms where h[idx] is the first factor: h[idx]*h[idx+lag] for lag where idx+lag < N
    - Terms where h[idx] is the second factor: h[idx-lag]*h[idx] for lag where idx-lag >= 0
    Each such term changes from val to -val, so delta = -2*val.
    """
    n = len(h)
    val = h[idx]
    # For each lag from -(N-1) to (N-1), but our c is indexed 0..2N-2 where lag = k - (N-1)
    # Actually c[k] corresponds to lag = k - (N-1) for k in 0..2N-2
    # Let's just compute the delta array directly.
    delta = np.zeros(2 * n - 1)
    # h[idx] as first factor: contributes to lags where j=idx, so lag = j2 - j = j2 - idx, j2 in [0,N-1]
    # In c array, index for lag L is L + (N-1)
    for j2 in range(n):
        lag = j2 - idx
        c_idx = lag + (n - 1)
        if 0 <= c_idx < len(delta):
            delta[c_idx] += -2 * val * h[j2]
    # h[idx] as second factor: contributes to lags where j2=idx, so lag = j2 - j = idx - j, j in [0,N-1]
    for j in range(n):
        lag = idx - j
        c_idx = lag + (n - 1)
        if 0 <= c_idx < len(delta):
            delta[c_idx] += -2 * val * h[j]
    return delta


def _hill_climb_fast(h, max_iter=500, deadline=140.0):
    """Fast local search using delta updates."""
    c = _fft_autoconv(h)
    max_abs, int_sq, ratio = _eval(h)
    s = float(np.sum(h))
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try a few random indices, pick the best
        best_idx = -1
        best_new_ratio = ratio
        best_new_c = None
        best_new_s = s
        
        for _ in range(5):
            idx = np.random.randint(N)
            delta = _delta_flip(c, h, idx)
            c_new = c + delta
            new_max = float(np.max(np.abs(c_new)))
            new_s = s - 2 * h[idx]
            new_int_sq = new_s * new_s
            if new_int_sq < 1e-12:
                new_ratio = new_max
            else:
                new_ratio = new_max / new_int_sq
            
            if new_ratio < best_new_ratio:
                best_new_ratio = new_ratio
                best_idx = idx
                best_new_c = c_new
                best_new_s = new_s
        
        if best_idx >= 0 and best_new_ratio < ratio:
            h[best_idx] = -h[best_idx]
            c = best_new_c
            s = best_new_s
            max_abs, int_sq, ratio = _eval(h)
    
    return h, ratio


def _build_rudin_shapiro(n):
    """Classic Rudin-Shapiro sequence of length >= n, truncated."""
    A = np.array([1.0])
    B = np.array([1.0])
    while len(A) < n:
        A, B = np.concatenate([A, B]), np.concatenate([A, -B])
    return A[:n]


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Strategy 1: Rudin-Shapiro
    h_rs = _build_rudin_shapiro(N)
    _, _, val_rs = _eval(h_rs)
    best_h = h_rs.copy()
    best_val = val_rs
    
    # Strategy 2: Multiple random starts with fast local search
    np.random.seed(42)
    for seed in range(100):
        if time.time() > deadline:
            break
        
        h_rand = np.random.choice([-1.0, 1.0], size=N)
        h_improved, val = _hill_climb_fast(h_rand, max_iter=200, deadline=deadline)
        
        if val < best_val:
            best_val = val
            best_h = h_improved.copy()
    
    # Strategy 3: Polish the best found
    if time.time() < deadline - 5:
        h_final, val_final = _hill_climb_fast(best_h.copy(), max_iter=500, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
