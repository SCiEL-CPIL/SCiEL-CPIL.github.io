import time
import numpy as np
import math
from itertools import combinations

_N = 24
_SQRN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 115.0  # Safety margin for 150s wall-clock

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
    Precompute DFT matrix columns.
    Returns Z_cols.
    """
    j = np.arange(M)
    k = np.arange(_N)
    J, K = np.meshgrid(j, k, indexing='ij')
    Z = np.exp(1j * 2 * np.pi * J * K / M)
    Z_cols = [Z[:, k] for k in range(_N)]
    return Z_cols

def _fast_local_search(coeffs, M=4096, max_iters=3000):
    """
    Local search using peak-tracked evaluation with single and double flips.
    coeffs: list of 24 ints
    """
    Z_cols = _setup_eval(M)
    coeffs_arr = np.array(coeffs, dtype=float)
    
    # Initial evaluation
    vals = np.zeros(M, dtype=complex)
    for k in range(_N):
        vals += coeffs_arr[k] * Z_cols[k]
    mags = np.abs(vals)
    current_max = np.max(mags)
    
    # Track top peak indices
    top_n = min(32, M)
    peak_idx = np.argsort(mags)[-top_n:]
    
    best_coeffs = coeffs[:]
    best_max = current_max
    no_improve_count = 0
    
    # Precompute all single and double flip deltas
    single_deltas = [(-2.0 * coeffs_arr[k]) * Z_cols[k] for k in range(_N)]
    double_deltas = {}
    for k1 in range(_N):
        for k2 in range(k1+1, _N):
            double_deltas[(k1, k2)] = (-2.0 * coeffs_arr[k1]) * Z_cols[k1] + (-2.0 * coeffs_arr[k2]) * Z_cols[k2]
    
    for it in range(max_iters):
        if _time_left() < 8.0:
            break
        
        best_new_max = current_max
        best_move = None
        
        # Evaluate at tracked peaks only
        peak_vals = vals[peak_idx]
        peak_mags = np.abs(peak_vals)
        
        # Try single flips
        for k in range(_N):
            new_peak_vals = peak_vals + single_deltas[k][peak_idx]
            new_max = np.max(np.abs(new_peak_vals))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_move = (k,)
        
        # Try double flips
        for (k1, k2), delta in double_deltas.items():
            new_peak_vals = peak_vals + delta[peak_idx]
            new_max = np.max(np.abs(new_peak_vals))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_move = (k1, k2)
        
        if best_move is not None:
            # Apply best move
            for k in best_move:
                coeffs_arr[k] = -coeffs_arr[k]
                vals += (-2.0 * coeffs_arr[k]) * Z_cols[k]
            
            # Update peak tracking
            mags = np.abs(vals)
            peak_idx = np.argsort(mags)[-top_n:]
            current_max = np.max(mags)
            
            best_coeffs = coeffs_arr.astype(int).tolist()
            best_max = current_max
            no_improve_count = 0
            
            # Update deltas for flipped coefficients
            for k in best_move:
                single_deltas[k] = (-2.0 * coeffs_arr[k]) * Z_cols[k]
                for k2 in range(_N):
                    if k2 != k:
                        if (k, k2) in double_deltas:
                            double_deltas[(k, k2)] = (-2.0 * coeffs_arr[k]) * Z_cols[k] + (-2.0 * coeffs_arr[k2]) * Z_cols[k2]
                        elif (k2, k) in double_deltas:
                            double_deltas[(k2, k)] = (-2.0 * coeffs_arr[k2]) * Z_cols[k2] + (-2.0 * coeffs_arr[k]) * Z_cols[k]
        else:
            no_improve_count += 1
            if no_improve_count > 15:
                break
            
    return best_coeffs, best_max

def _eval_precise(coeffs, M=65536):
    """
    High-precision evaluation with golden-section refinement.
    """
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:_N] = coeffs
    vals = np.fft.ifft(c_padded) * M
    mags = np.abs(vals)
    
    # Find top 30 peaks
    top_idx = np.argsort(mags)[-30:]
    max_val = float(mags[top_idx[0]])
    
    k = np.arange(_N)
    
    def P_mag(theta):
        zk = np.exp(1j * theta * k)
        return float(np.abs(np.sum(coeffs * zk)))
        
    gr = (math.sqrt(5) - 1) / 2
    
    for idx in top_idx:
        theta0 = 2 * np.pi * idx / M
        a = theta0 - np.pi / M
        b = theta0 + np.pi / M
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = P_mag(c)
        fd = P_mag(d)
        for _ in range(50):
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
    np.random.seed(42)
    
    # Initial candidate: Rudin-Shapiro
    best_coeffs = _rs_poly()
    best_val = _eval_precise(best_coeffs)
    
    # Local search from RS
    c1, v1 = _fast_local_search(best_coeffs, M=4096, max_iters=2000)
    if v1 < best_val:
        best_coeffs = c1
        best_val = v1
    
    # Random restarts
    while _time_left() > 20.0:
        if np.random.rand() < 0.4:
            c_rand = best_coeffs[:]
            n_flips = np.random.randint(1, 6)
            for _ in range(n_flips):
                c_rand[np.random.randint(_N)] *= -1
        else:
            c_rand = [1 if np.random.rand() > 0.5 else -1 for _ in range(_N)]
        
        c2, v2 = _fast_local_search(c_rand, M=2048, max_iters=1500)
        
        if v2 < best_val:
            v2_precise = _eval_precise(c2)
            if v2_precise < best_val:
                best_coeffs = c2
                best_val = v2_precise
    
    # Final polish
    if _time_left() > 12.0:
        c3, v3 = _fast_local_search(best_coeffs, M=8192, max_iters=1000)
        if v3 < best_val:
            v3_precise = _eval_precise(c3)
            if v3_precise < best_val:
                best_coeffs = c3
                best_val = v3_precise

    return best_coeffs
