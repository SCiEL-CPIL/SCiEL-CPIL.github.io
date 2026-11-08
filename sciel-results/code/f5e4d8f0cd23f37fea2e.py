import time
import numpy as np
import scipy.signal

N = 400

def _autoconv_max_abs(h):
    c = np.convolve(h, h, mode='full')
    return float(np.max(np.abs(c)))

def _integral_sq(h):
    s = float(np.sum(h))
    return s * s

def _evaluate(h):
    c = np.convolve(h, h, mode='full')
    max_abs = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    s2 = s * s
    if s2 < 1e-12:
        return max_abs
    return max_abs / s2

def _fast_flip_delta(c, h, idx, N):
    s = h[idx]
    delta = np.zeros(2 * N - 1)
    i1_start = max(0, N - 1 - idx)
    i1_end = min(2 * N - 2, 2 * N - 2 - idx)
    if i1_start <= i1_end:
        lags1 = np.arange(i1_start, i1_end + 1) - (N - 1)
        idxs1 = idx + lags1
        mask = (lags1 != 0)
        if np.any(mask):
            delta[i1_start:i1_end + 1][mask] = -2 * s * h[idxs1[mask]]
    
    i2_start = max(0, idx)
    i2_end = min(2 * N - 2, N - 1 + idx)
    if i2_start <= i2_end:
        lags2 = np.arange(i2_start, i2_end + 1) - (N - 1)
        idxs2 = idx - lags2
        mask = (lags2 != 0)
        if np.any(mask):
            delta[i2_start:i2_end + 1][mask] += -2 * s * h[idxs2[mask]]
    
    return delta

def _hill_climb_fast(h, max_iter=5000, deadline=140.0):
    c = np.convolve(h, h, mode='full')
    current_max = float(np.max(np.abs(c)))
    current_sum = float(np.sum(h))
    current_s2 = current_sum * current_sum
    current_val = current_max / current_s2 if current_s2 > 1e-12 else current_max
    
    indices = np.arange(N)
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        delta = _fast_flip_delta(c, h, idx, N)
        c_new = c + delta
        
        new_max = float(np.max(np.abs(c_new)))
        new_sum = current_sum - 2 * h[idx]
        new_s2 = new_sum * new_sum
        new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max
        
        if new_val < current_val:
            h[idx] = -h[idx]
            c = c_new
            current_max = new_max
            current_sum = new_sum
            current_s2 = new_s2
            current_val = new_val
    
    return h, current_val, c

def _build_spectral_init(n, rng):
    # Generate random spectrum
    spectrum = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    # Ensure Hermitian symmetry for real signal
    spectrum[0] = spectrum[0].real
    if n % 2 == 0:
        spectrum[n//2] = spectrum[n//2].real
    else:
        spectrum[1:n//2+1] = spectrum[1:n//2+1]
        spectrum[n//2+1:] = spectrum[1:n//2+1][::-1].conj()
    
    # Inverse FFT
    h = np.fft.ifft(spectrum).real
    # Normalize to sum=1 to avoid zero integral issues, or just use as is
    # The objective is scale invariant, so we can just use the raw values
    # But to avoid numerical issues with sum=0, we can add a small DC component
    # Or simply use the values and let the optimizer handle it
    return h

def construct():
    deadline = 140.0
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(123)
    
    # Try a few spectral initializations
    for i in range(5):
        if time.time() > deadline - 10:
            break
            
        h = _build_spectral_init(N, rng)
        # Scale to have sum 1? Or just use it.
        # Let's try to optimize it directly.
        h_opt, val, _ = _hill_climb_fast(h.copy(), max_iter=2000, deadline=deadline - 5)
        
        if val < best_val:
            best_val = val
            best_h = h_opt
    
    # Final polish with constant sequence if it's better
    const_val = _evaluate(np.ones(N))
    if const_val < best_val:
        best_h = np.ones(N)
        best_val = const_val
        
    return best_h
