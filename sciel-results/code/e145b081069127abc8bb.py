import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 150.0
# Use a safety margin to ensure we return before the hard limit
_TIME_MARGIN = 15.0

def _time_left():
    return _LIMIT - _TIME_MARGIN - (time.time() - _T0)

def _eval_peak_sq(coeffs, Z, theta, M):
    """Evaluate the maximum |P(z)|^2 on the grid for given coefficients."""
    vals = coeffs @ Z.T
    mag2 = np.abs(vals) ** 2
    peak_sq = np.max(mag2)
    peak_idx = np.argmax(mag2)
    return peak_sq, peak_idx

def _refine_peak(coeffs, peak_val_sq, phase, M_grid=512, iters=20):
    """
    Refine the peak around a specific phase using golden section search.
    coeffs: array of 24 floats (+1/-1)
    peak_val_sq: current known peak magnitude squared (for comparison)
    phase: initial phase estimate in radians
    Returns refined peak magnitude squared.
    """
    c_arr = coeffs
    powers = np.arange(_N)
    
    def f(a):
        z = np.exp(1j * a)
        zk = np.power(z, powers)
        s = np.dot(c_arr, zk)
        return (s.real * s.real + s.imag * s.imag)

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
    return max(peak2, peak_val_sq)

def construct():
    global _T0
    _T0 = time.time()
    
    # Brute force approach over all 2^24 = 16,777,216 candidates
    M = 512
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    
    best_peak_sq = float('inf')
    best_coeff_idx = 0
    best_peak_phase_idx = 0
    
    # Use smaller batches to manage memory
    B = 5000
    total = 1 << _N
    
    # Precompute bit masks
    bit_mask = 1 << np.arange(_N, dtype=np.uint32)
    
    for start in range(0, total, B):
        if _time_left() < 3.0:
            break
        
        end = min(start + B, total)
        idxs = np.arange(start, end, dtype=np.uint32)
        bits = (idxs[:, None] & bit_mask[None, :]) > 0
        coeffs = bits.astype(np.float64) * 2.0 - 1.0
        
        # Evaluate all candidates in this batch
        vals = coeffs @ Z.T
        mag2 = np.abs(vals) ** 2
        peaks = np.max(mag2, axis=1)
        
        min_idx = np.argmin(peaks)
        min_peak_sq = peaks[min_idx]
        
        if min_peak_sq < best_peak_sq:
            best_peak_sq = min_peak_sq
            best_coeff_idx = start + min_idx
            best_peak_phase_idx = np.argmax(mag2[min_idx])
            best_peak_phase = theta[best_peak_phase_idx]
            
    # Retrieve best coefficients
    bits = (best_coeff_idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
    best_coeffs = bits.astype(np.float64) * 2.0 - 1.0
    
    # Refine the peak of the best candidate
    true_peak_sq = _refine_peak(best_coeffs, best_peak_sq, best_peak_phase, M_grid=M)
    
    # Local search: try flipping pairs of bits to escape local minima
    cur_coeffs = best_coeffs.copy()
    cur_peak_sq = true_peak_sq
    
    # Try all single flips first
    for j in range(_N):
        if _time_left() < 2.0:
            break
        cur_coeffs[j] = -cur_coeffs[j]
        new_peak_sq, new_peak_idx = _eval_peak_sq(cur_coeffs, Z, theta, M)
        if new_peak_sq < cur_peak_sq:
            # Refine to get true peak
            new_phase = theta[new_peak_idx]
            refined_sq = _refine_peak(cur_coeffs, new_peak_sq, new_phase, M_grid=M)
            if refined_sq < cur_peak_sq:
                cur_peak_sq = refined_sq
                best_coeffs = cur_coeffs.copy()
                best_peak_sq = cur_peak_sq
            else:
                cur_coeffs[j] = -cur_coeffs[j]
        else:
            cur_coeffs[j] = -cur_coeffs[j]
    
    # Try double flips if time permits
    if _time_left() > 5.0:
        for j in range(_N):
            if _time_left() < 2.0:
                break
            for k2 in range(j + 1, _N):
                if _time_left() < 1.0:
                    break
                cur_coeffs[j] = -cur_coeffs[j]
                cur_coeffs[k2] = -cur_coeffs[k2]
                new_peak_sq, new_peak_idx = _eval_peak_sq(cur_coeffs, Z, theta, M)
                if new_peak_sq < cur_peak_sq:
                    new_phase = theta[new_peak_idx]
                    refined_sq = _refine_peak(cur_coeffs, new_peak_sq, new_phase, M_grid=M)
                    if refined_sq < cur_peak_sq:
                        cur_peak_sq = refined_sq
                        best_coeffs = cur_coeffs.copy()
                        best_peak_sq = cur_peak_sq
                        # Restart single flips from here
                        break
                cur_coeffs[j] = -cur_coeffs[j]
                cur_coeffs[k2] = -cur_coeffs[k2]
            else:
                continue
            break
    
    # Convert back to list of ints
    result = [int(1 if c > 0 else -1) for c in best_coeffs]
    
    return result
