import time
import numpy as np
import math

_N = 24
_SQRN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 115.0 # Safety margin for 120s limit

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
    Returns Z (M x N) such that vals = Z @ coeffs gives P(z_j).
    """
    j = np.arange(M)
    k = np.arange(_N)
    J, K = np.meshgrid(j, k, indexing='ij')
    Z = np.exp(1j * 2 * np.pi * J * K / M)
    return Z

# Precompute for fast search
_Z_FAST = _setup_eval(4096)
_M_FAST = 4096

def _fast_local_search(coeffs, max_iters=3000):
    """
    Local search using incremental FFT updates with single and double flips.
    coeffs: list of 24 ints
    """
    Z = _Z_FAST
    coeffs_arr = np.array(coeffs, dtype=float)
    
    # Initial evaluation
    vals = Z @ coeffs_arr
    mags = np.abs(vals)
    current_max = np.max(mags)
    
    best_coeffs = coeffs[:]
    best_max = current_max
    no_improve_count = 0
    
    for it in range(max_iters):
        if _time_left() < 10.0:
            break
        
        improved = False
        best_flip_idx = -1
        best_flip2_idx = -1
        best_new_max = current_max
        
        # Try single flips
        for k in range(_N):
            old_c = coeffs_arr[k]
            delta = -2.0 * old_c * Z[:, k]
            new_vals = vals + delta
            new_max = np.max(np.abs(new_vals))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip_idx = k
        
        # Try double flips (only if no single flip improved or to find better double)
        # Optimization: Only try doubles if singles didn't improve significantly or 
        # we want to escape local minima. 
        # For efficiency, let's try doubles only if no single flip was found, 
        # or periodically.
        if best_flip_idx == -1:
            for k1 in range(_N):
                for k2 in range(k1 + 1, _N):
                    old_c1 = coeffs_arr[k1]
                    old_c2 = coeffs_arr[k2]
                    delta = -2.0 * old_c1 * Z[:, k1] - 2.0 * old_c2 * Z[:, k2]
                    new_vals = vals + delta
                    new_max = np.max(np.abs(new_vals))
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_flip_idx = k1
                        best_flip2_idx = k2
        
        if best_flip_idx != -1:
            # Apply best flip(s)
            k = best_flip_idx
            old_c = coeffs_arr[k]
            coeffs_arr[k] = -old_c
            vals += -2.0 * old_c * Z[:, k]
            
            if best_flip2_idx != -1:
                k2 = best_flip2_idx
                old_c2 = coeffs_arr[k2]
                coeffs_arr[k2] = -old_c2
                vals += -2.0 * old_c2 * Z[:, k2]
            
            current_max = best_new_max
            best_coeffs = coeffs_arr.astype(int).tolist()
            best_max = current_max
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
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
    
    # Find top 10 peaks
    top_idx = np.argsort(mags)[-10:]
    max_val = mags[top_idx[0]]
    
    # Precompute powers for faster evaluation
    k = np.arange(_N)
    
    def P_mag(theta):
        z = np.exp(1j * theta)
        zk = np.exp(1j * theta * k)
        return np.abs(np.sum(coeffs * zk))
        
    for idx in top_idx:
        theta0 = 2 * np.pi * idx / M
        a = theta0 - np.pi / M
        b = theta0 + np.pi / M
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = P_mag(c)
        fd = P_mag(d)
        for _ in range(35):
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

def _precise_local_search(coeffs, max_iters=10):
    """
    Local search using precise evaluation.
    """
    coeffs = list(coeffs)
    current_val = _eval_precise(coeffs)
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
            
        best_val = current_val
        best_coeffs = coeffs[:]
        
        # Try single flips
        for k in range(_N):
            old_c = coeffs[k]
            coeffs[k] = -old_c
            new_val = _eval_precise(coeffs)
            
            if new_val < best_val:
                best_val = new_val
                best_coeffs = coeffs[:]
            else:
                coeffs[k] = old_c
                
        if best_val < current_val:
            current_val = best_val
            coeffs = best_coeffs
        else:
            break
            
    return coeffs, current_val

def construct():
    t_start = time.time()
    np.random.seed(42)
    
    # Initial candidate: Rudin-Shapiro
    best_coeffs = _rs_poly()
    best_val = _eval_precise(best_coeffs)
    
    candidates = []
    
    # 1. Fast local search from RS
    c1, v1 = _fast_local_search(best_coeffs, max_iters=2000)
    candidates.append((v1, c1))
    
    # 2. Fast local search from random starts
    # We will collect candidates and then polish the best ones
    num_restarts = 20
    for i in range(num_restarts):
        if _time_left() < 20.0:
            break
            
        # Mix of random and perturbed RS
        if i % 3 == 0 and best_coeffs:
            # Perturb best found so far
            c_rand = best_coeffs[:]
            for _ in range(np.random.randint(1, 4)):
                c_rand[np.random.randint(_N)] *= -1
        else:
            c_rand = [1 if np.random.rand() > 0.5 else -1 for _ in range(_N)]
        
        c2, v2 = _fast_local_search(c_rand, max_iters=1000)
        candidates.append((v2, c2))
        
        # Keep track of the best fast candidate for potential perturbation
        if v2 < v1:
            v1 = v2
            c1 = c2
            
    # Sort candidates by fast score
    candidates.sort(key=lambda x: x[0])
    
    # 3. Precise polish on the top candidates
    # We have ~20 candidates. Let's polish the top 10.
    top_candidates = candidates[:10]
    
    for fast_score, cands in top_candidates:
        if _time_left() < 10.0:
            break
            
        # Polish this candidate
        c_polished, v_polished = _precise_local_search(cands, max_iters=5)
        
        if v_polished < best_val:
            best_val = v_polished
            best_coeffs = c_polished
            
    # 4. Final check: if we have time, try to polish the global best again
    if _time_left() > 5.0:
        c_final, v_final = _precise_local_search(best_coeffs, max_iters=3)
        if v_final < best_val:
            best_coeffs = c_final
            best_val = v_final

    return best_coeffs
