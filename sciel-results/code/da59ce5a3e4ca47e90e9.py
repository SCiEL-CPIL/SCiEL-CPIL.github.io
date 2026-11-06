import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0
_TIME_MARGIN = 15.0

def _time_left():
    return _LIMIT - _TIME_MARGIN - (time.time() - _T0)

def _eval_poly(coeffs, theta):
    """Evaluate polynomial at multiple angles using FFT-like approach."""
    # coeffs: (24,) array of +1/-1
    # theta: (M,) array of angles
    # Returns: (M,) complex values P(e^{i*theta})
    z = np.exp(1j * theta)
    powers = np.arange(_N)
    zk = np.power(z[:, None], powers[None, :])  # (M, 24)
    return zk @ coeffs  # (M,)

def _refine_peak(coeffs, peak_val_sq, phase, iters=20):
    """
    Refine the peak around a specific phase using golden section search.
    coeffs: array of 24 floats (+1/-1)
    peak_val_sq: current known peak magnitude squared
    phase: initial phase estimate in radians
    Returns: refined peak magnitude (not squared)
    """
    def f(a):
        z = np.exp(1j * a)
        zk = z ** np.arange(_N)
        s = np.dot(coeffs, zk)
        return s.real * s.real + s.imag * s.imag

    M_grid = 256
    d_phase = 2.0 * math.pi / M_grid
    lo = phase - d_phase
    hi = phase + d_phase
    
    f_lo = f(lo)
    f_hi = f(hi)
    
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    
    for _ in range(iters):
        m1 = hi - gr * (hi - lo)
        m2 = lo + gr * (hi - lo)
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            hi = m2
            f_hi = f2
        else:
            lo = m1
            f_lo = f1
            
    peak2 = max(f_lo, f_hi)
    return math.sqrt(max(peak2, peak_val_sq))

def _full_peak_eval(coeffs, M=512):
    """Evaluate max |P(z)| over dense grid and refine top peaks."""
    theta = 2.0 * np.pi * np.arange(M) / M
    vals = _eval_poly(coeffs, theta)
    mag2 = np.abs(vals) ** 2
    peak_idx = np.argmax(mag2)
    peak_sq = mag2[peak_idx]
    phase = theta[peak_idx]
    refined = _refine_peak(coeffs, peak_sq, phase)
    return refined * refined

def construct():
    global _T0
    _T0 = time.time()
    
    # Use a coarser grid for brute force to be faster, then refine
    M_coarse = 256
    theta = 2.0 * np.pi * np.arange(M_coarse) / M_coarse
    z = np.exp(1j * theta)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    
    best_peak_sq = float('inf')
    best_coeff_idx = 0
    
    B = 50000
    total = 1 << _N
    
    for start in range(0, total, B):
        if _time_left() < 10.0:
            break
        
        end = min(start + B, total)
        idxs = np.arange(start, end, dtype=np.uint32)
        bit_mask = 1 << np.arange(_N, dtype=np.uint32)
        bits = (idxs[:, None] & bit_mask[None, :]) > 0
        coeffs = (bits.astype(np.float64) * 2.0) - 1.0
        
        vals = coeffs @ Z.T
        mag2 = np.abs(vals) ** 2
        peaks = np.max(mag2, axis=1)
        
        min_idx = np.argmin(peaks)
        min_peak_sq = peaks[min_idx]
        
        if min_peak_sq < best_peak_sq:
            best_peak_sq = min_peak_sq
            best_coeff_idx = start + min_idx
            
    # Retrieve best coefficients
    bits = (best_coeff_idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
    best_coeffs = (bits.astype(np.float64) * 2.0) - 1.0
    
    # Refine the peak with high precision
    cur_peak_sq = _full_peak_eval(best_coeffs, M=512)
    best_coeffs = best_coeffs.copy()
    
    # Local search: try flipping bits, keep if improvement
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for j in range(_N):
            if _time_left() < 3.0:
                break
            best_coeffs[j] = -best_coeffs[j]
            new_peak_sq = _full_peak_eval(best_coeffs, M=256)
            if new_peak_sq < cur_peak_sq:
                cur_peak_sq = new_peak_sq
                improved = True
            else:
                best_coeffs[j] = -best_coeffs[j]
    
    result = [int(1.0 if c > 0 else -1.0) for c in best_coeffs]
    return result
