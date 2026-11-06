import time
import numpy as np
import math

_N = 24
_SQRN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 115.0 # Safety margin

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly():
    """Generate Rudin-Shapiro coefficients truncated to _N."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]

def _setup_eval(M=4096):
    """
    Precompute DFT matrix columns for fast incremental updates.
    Returns Z (M x N) and a function to evaluate.
    """
    # We need Z such that vals = Z @ coeffs
    # vals[j] = sum_k coeffs[k] * exp(i * 2 * pi * j * k / M)
    # This is the Inverse DFT scaled by M? 
    # np.fft.ifft(c_padded) * M gives sum c_k exp(i 2pi j k / M).
    # So Z[j, k] = exp(i * 2 * pi * j * k / M).
    
    j = np.arange(M)
    k = np.arange(_N)
    # Create meshgrid
    J, K = np.meshgrid(j, k, indexing='ij')
    Z = np.exp(1j * 2 * np.pi * J * K / M)
    
    return Z

def _fast_local_search(coeffs, M=2048, max_iters=5000):
    """
    Local search using incremental FFT updates.
    coeffs: list of 24 ints
    """
    Z = _setup_eval(M)
    coeffs_arr = np.array(coeffs, dtype=float)
    
    # Initial evaluation
    vals = Z @ coeffs_arr
    mags = np.abs(vals)
    current_max = np.max(mags)
    
    best_coeffs = coeffs[:]
    best_max = current_max
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        improved = False
        best_flip_idx = -1
        best_new_max = current_max
        
        # Try single flips
        for k in range(_N):
            # Flip k
            # New vals = vals - 2 * coeffs[k] * Z[:, k]
            delta = -2.0 * coeffs_arr[k] * Z[:, k]
            new_vals = vals + delta
            new_max = np.max(np.abs(new_vals))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip_idx = k
        
        if best_flip_idx != -1:
            # Apply best flip
            k = best_flip_idx
            coeffs_arr[k] *= -1
            vals += -2.0 * coeffs_arr[k] * -1 * Z[:, k] # Wait, delta was calculated based on old coeffs.
            # Let's just recompute or update carefully.
            # old_c = coeffs_arr[k]
            # new_c = -old_c
            # delta = (new_c - old_c) * Z[:, k] = -2 * old_c * Z[:, k]
            # We already added delta to vals in the check? No, we only checked.
            # So we must apply the update now.
            
            old_c = coeffs_arr[k]
            coeffs_arr[k] = -old_c
            # Update vals
            vals += -2.0 * old_c * Z[:, k]
            current_max = best_new_max
            best_coeffs = coeffs_arr.astype(int).tolist()
            best_max = current_max
            improved = True
        else:
            # Try double flips? 
            # To keep it simple and fast, maybe just break and rely on restarts.
            # Or implement double flips if time permits.
            # For now, single flip local search.
            break
            
    return best_coeffs, best_max

def _eval_precise(coeffs, M=16384):
    """
    More precise evaluation for final comparison.
    """
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:_N] = coeffs
    vals = np.fft.ifft(c_padded) * M
    mags = np.abs(vals)
    
    # Find top 5 peaks
    top_idx = np.argsort(mags)[-5:]
    max_val = mags[top_idx[0]]
    
    # Refine peaks
    def P_val(theta):
        z = np.exp(1j * theta)
        k = np.arange(_N)
        zk = z ** k
        return np.sum(coeffs * zk)
        
    def P_mag(theta):
        return np.abs(P_val(theta))
        
    for idx in top_idx:
        theta0 = 2 * np.pi * idx / M
        a = theta0 - np.pi / M
        b = theta0 + np.pi / M
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = P_mag(c)
        fd = P_mag(d)
        for _ in range(30):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = P_mag(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = P_mag(d)
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def construct():
    t_start = time.time()
    
    # Initial candidate: Rudin-Shapiro
    best_coeffs = _rs_poly()
    best_val = _eval_precise(best_coeffs)
    
    # Strategy:
    # 1. Run fast local search from RS.
    # 2. Run fast local search from random starts.
    
    # Local search 1: RS
    c1, v1 = _fast_local_search(best_coeffs, M=2048, max_iters=5000)
    if v1 < best_val:
        best_coeffs = c1
        best_val = v1
        
    # Random restarts
    while _time_left() > 10.0:
        c_rand = [1 if np.random.rand() > 0.5 else -1 for _ in range(_N)]
        c2, v2 = _fast_local_search(c_rand, M=1024, max_iters=2000)
        
        if v2 < best_val:
            # Verify with higher precision
            v2_precise = _eval_precise(c2)
            if v2_precise < best_val:
                best_coeffs = c2
                best_val = v2_precise
                
    # Final polish
    if _time_left() > 5.0:
        c3, v3 = _fast_local_search(best_coeffs, M=4096, max_iters=2000)
        if v3 < best_val:
             v3_precise = _eval_precise(c3)
             if v3_precise < best_val:
                 best_coeffs = c3
                 best_val = v3_precise

    return best_coeffs
