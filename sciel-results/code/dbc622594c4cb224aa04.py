"""
Combined program: Golay-Rudin-Shapiro seed + NumPy FFT evaluation + iterative local search
with golden-section refinement, multiple random restarts, and time-budget management.
"""
import time
import math
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_LIMIT = 145.0  # safety margin before 150s kill

def _time_left():
    return _LIMIT - (time.time() - time.time())  # placeholder, actual tracking below

def _rs_poly(n):
    """Generate Rudin-Shapiro coefficients of length n."""
    R = [1]
    S = [1]
    while len(R) < n:
        new_R = R + S
        new_S = R + [-s for s in S]
        R = new_R
        S = new_S
    return R[:n]

def _calc_peak(coeffs, M=4096, refine_top=10, refine_iters=40):
    """
    Calculate the peak magnitude of P(z) on the unit circle.
    Uses dense sampling via FFT and golden-section refinement for top peaks.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    # Find indices of top peaks and refine with golden section
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        powers_arr = powers
        
        for idx in top_indices:
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers_arr)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        peak_val = best_peak
    
    return math.sqrt(peak_val)

def _calc_peak_fast(coeffs, M=2048, refine_top=5, refine_iters=20):
    """
    Faster peak calculation for local search iterations.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        powers_arr = powers
        
        for idx in top_indices:
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers_arr)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        peak_val = best_peak
    
    return math.sqrt(peak_val)

def _local_search(seed, start_time, max_time, M=2048):
    """
    Perform local search to minimize the peak.
    Uses fast peak calculation for speed, with periodic high-accuracy checks.
    """
    cur = list(seed)
    cur_peak = _calc_peak_fast(cur, M, refine_top=5, refine_iters=20)
    best = list(cur)
    best_peak = cur_peak
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Single flip pass
        for j in range(_N):
            if time.time() - start_time > max_time:
                break
            cur[j] = -cur[j]
            p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=15)
            if p < cur_peak - 1e-12:
                cur_peak = p
                improved = True
                if cur_peak < best_peak - 1e-12:
                    best = list(cur)
                    best_peak = cur_peak
            else:
                cur[j] = -cur[j]
        
        if not improved:
            # Try double flips
            double_improved = False
            for j in range(_N):
                if time.time() - start_time > max_time:
                    break
                for k in range(j + 1, _N):
                    if time.time() - start_time > max_time:
                        break
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
                    p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=15)
                    if p < cur_peak - 1e-12:
                        cur_peak = p
                        double_improved = True
                        if cur_peak < best_peak - 1e-12:
                            best = list(cur)
                            best_peak = cur_peak
                    else:
                        cur[j] = -cur[j]
                        cur[k] = -cur[k]
                
                if time.time() - start_time > max_time:
                    break
            
            if not double_improved:
                # Try triple flips if stuck
                triple_improved = False
                for j in range(_N):
                    if time.time() - start_time > max_time:
                        break
                    for k in range(j + 1, _N):
                        if time.time() - start_time > max_time:
                            break
                        for l in range(k + 1, _N):
                            if time.time() - start_time > max_time:
                                break
                            cur[j] = -cur[j]
                            cur[k] = -cur[k]
                            cur[l] = -cur[l]
                            p = _calc_peak_fast(cur, M, refine_top=3, refine_iters=15)
                            if p < cur_peak - 1e-12:
                                cur_peak = p
                                triple_improved = True
                                if cur_peak < best_peak - 1e-12:
                                    best = list(cur)
                                    best_peak = cur_peak
                            else:
                                cur[j] = -cur[j]
                                cur[k] = -cur[k]
                                cur[l] = -cur[l]
                        
                        if time.time() - start_time > max_time:
                            break
                    if time.time() - start_time > max_time:
                        break
                
                if not triple_improved:
                    break
                else:
                    improved = True
            else:
                improved = True
        
        if not improved:
            break
    
    return best, best_peak

def construct():
    start_time = time.time()
    deadline = _LIMIT
    
    # Golay-Rudin-Shapiro construction for 24 coefficients
    coeffs = _rs_poly(24)
    
    # First, do a local search from the GR seed
    best_coeffs, best_score = _local_search(coeffs, start_time, deadline - 20)
    
    # Now try multiple random restarts to escape local minima
    rng = np.random.RandomState(42)
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() - start_time > deadline - 10:
            break
        
        # Generate a random starting point
        random_coeffs = [1 if rng.randint(0, 2) == 0 else -1 for _ in range(_N)]
        
        # Do a quick local search from this random start
        local_best, local_score = _local_search(random_coeffs, start_time, deadline - 5)
        
        if local_score < best_score - 1e-12:
            best_coeffs = local_best
            best_score = local_score
    
    # Final high-accuracy verification
    final_peak = _calc_peak(best_coeffs, M=8192, refine_top=10, refine_iters=50)
    
    return best_coeffs
