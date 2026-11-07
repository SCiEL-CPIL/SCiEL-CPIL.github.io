import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
w = 1.0 / 100.0
m = 2 * N - 1  # 99
m_plus_1 = m + 1  # 100
_x = np.arange(N)
_c = (N - 1) / 2.0

def compute_value(h):
    """
    Compute the objective value for given step heights h.
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    # FFT based convolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
        
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
    g = np.real(g)
    
    # Clip small negative values from numerical noise
    g = np.maximum(g, 0.0)
    
    # Calculate components
    sum_g_sq = np.sum(g**2)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)
    max_g = np.max(g)
    
    if max_g == 0 or sum_g == 0:
        return 0.0
        
    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * max_g
    
    return numerator / denominator

def compute_value_smooth(h, beta=10.0):
    """
    Compute the objective value with log-sum-exp smoothed max for better gradient flow.
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    # FFT based convolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
        
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
    g = np.real(g)
    
    g = np.maximum(g, 0.0)
    
    # Calculate components
    sum_g_sq = np.sum(g**2)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)
    
    # Log-sum-exp smoothed max
    max_g = np.max(g)
    if max_g > 0:
        scaled = beta * (g - max_g)
        scaled = np.clip(scaled, -50, 50)
        smooth_max = max_g + (1.0 / beta) * np.log(np.sum(np.exp(scaled)))
    else:
        smooth_max = max_g
    
    if smooth_max <= 0 or sum_g <= 0:
        return 0.0
        
    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * smooth_max
    
    return numerator / denominator

def neg_value(log_h):
    """
    Objective function for minimization in log-space.
    """
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value(h)

def neg_value_smooth(log_h, beta=10.0):
    """
    Objective function for minimization in log-space with smoothed max.
    """
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value_smooth(h, beta)

def parametric_h_gaussian(params):
    sigma = np.exp(params[0])
    x = _x
    c = _c
    h = np.exp(-((x - c)**2) / (2.0 * sigma**2))
    return h

def parametric_h_super_gaussian(params):
    sigma = np.exp(params[0])
    beta = params[1]
    x = _x
    c = _c
    dist = np.abs(x - c)
    h = np.exp(-(dist**beta) / (2.0 * (sigma**beta)))
    return h

def parametric_h_polynomial(params):
    R = np.exp(params[0])
    p = params[1]
    x = _x
    c = _c
    t = (x - c) / R
    h = np.maximum(1.0 - t**2, 0.0)
    h = h**p
    return h

def parametric_h_flat_top(params):
    w_width = np.exp(params[0])
    x = _x
    c = _c
    d = np.abs(x - c)
    half_w = w_width / 2.0
    h = np.zeros(N)
    mask_inner = d < half_w
    mask_edge = (d >= half_w) & (d <= w_width)
    h[mask_inner] = 1.0
    d_edge = d[mask_edge]
    t = (d_edge - half_w) / (w_width - half_w)
    h[mask_edge] = 0.5 * (1.0 + np.cos(np.pi * t))
    return h

def parametric_h_hamming(params):
    R = np.exp(params[0])
    x = _x
    c = _c
    d = np.abs(x - c)
    h = np.zeros(N)
    mask = d <= R
    if R > 0:
        t = d[mask] / R
        h[mask] = 0.54 - 0.46 * np.cos(np.pi * t)
    return h

def parametric_h_blackman(params):
    R = np.exp(params[0])
    x = _x
    c = _c
    d = np.abs(x - c)
    h = np.zeros(N)
    mask = d <= R
    if R > 0:
        t = d[mask] / R
        h[mask] = 0.42 - 0.5 * np.cos(np.pi * t) + 0.08 * np.cos(2.0 * np.pi * t)
    return h

def parametric_h_two_gauss(params):
    s = np.exp(params[0])
    a = np.exp(params[1])
    x = _x
    c = _c
    h = np.exp(-((x - c - a)**2) / (2.0 * s**2)) + np.exp(-((x - c + a)**2) / (2.0 * s**2))
    return h

def parametric_h_dct(params):
    """
    h = softplus(DCT-II(c) + offset), where c is a vector of spectral coefficients.
    This ensures h >= 0 by construction.
    """
    c = params[:N]
    offset = params[N]
    # DCT-II via FFT
    # Pad c to 2N with zeros
    padded = np.zeros(2 * N)
    padded[:N] = c
    # Apply DCT-II formula using FFT
    # DCT-II: Y_k = 2 * sum_{n=0}^{N-1} x_n cos(pi*(2n+1)*k/(2N))
    # This can be computed via FFT of a modified signal
    # Simpler: use scipy.fft.dct for speed? No, stick to numpy for portability
    # Actually scipy.fft is available and fast.
    try:
        from scipy.fft import dct
        h_raw = dct(c, type=2, norm='ortho')
    except ImportError:
        # Fallback to numpy if scipy.fft not available (should be)
        # This is a slow fallback, just in case
        k = np.arange(N)[:, None]
        n = np.arange(N)[None, :]
        cos_arg = np.pi * (2 * n + 1) * k / (2 * N)
        h_raw = 2 * (c[None, :] * np.cos(cos_arg)).sum(axis=1)
        h_raw = h_raw * np.sqrt(1.0 / N) # Approximate norm
        h_raw[0] = c.mean() # DC term
    
    h = np.logaddexp(0, h_raw + offset) # softplus
    return h

def optimize_parametric(func, bounds, initial_guess=None):
    """
    Use differential_evolution for global optimization of parameters.
    """
    def obj(params):
        try:
            h = func(params)
            if np.any(h < 0) or np.any(~np.isfinite(h)) or np.max(h) == 0:
                return 1e6
            return -compute_value(h)
        except:
            return 1e6
            
    try:
        res = differential_evolution(obj, bounds, seed=42, maxiter=100, tol=1e-6, polish=True, popsize=15)
        best_params = res.x
        best_h = func(best_params)
        best_val = compute_value(best_h)
        return best_val, best_h, best_params
    except Exception:
        return 0.0, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer
    
    # Store best solution found
    best_val = 0.0
    best_h = np.ones(N)
    seed_val = compute_value(best_h)
    if seed_val > best_val:
        best_val = seed_val
        best_h = best_h.copy()
        
    # List of candidate h vectors for local refinement
    candidates = []
    
    # 1. Seed
    candidates.append(np.ones(N))
    
    # 2. Parametric Global Optimization
    families = [
        ("gauss", parametric_h_gaussian, [[0.0, 3.0]]),
        ("super_gauss", parametric_h_super_gaussian, [[0.0, 3.0], [1.0, 5.0]]),
        ("poly", parametric_h_polynomial, [[1.5, 3.5], [1.0, 5.0]]),
        ("flat_top", parametric_h_flat_top, [[2.0, 4.0]]),
        ("hamming", parametric_h_hamming, [[1.5, 3.5]]),
        ("blackman", parametric_h_blackman, [[1.5, 3.5]]),
        ("two_gauss", parametric_h_two_gauss, [[0.5, 2.5], [0.5, 2.5]]),
    ]
    
    for name, func, bounds in families:
        if time.time() > deadline - 50.0:
            break
        val, h, _ = optimize_parametric(func, bounds)
        if h is not None and val > best_val:
            best_val = val
            best_h = h.copy()
        if h is not None:
            candidates.append(h)
            
    # 3. DCT-based parametric search
    if time.time() < deadline - 40.0:
        # Optimize DCT coefficients directly
        # Bounds for DCT coefficients: [-2, 2] for each, offset [-2, 2]
        dct_bounds = [(-2.0, 2.0)] * N + [(-2.0, 2.0)]
        val, h, _ = optimize_parametric(parametric_h_dct, dct_bounds)
        if h is not None and val > best_val:
            best_val = val
            best_h = h.copy()
        if h is not None:
            candidates.append(h)

    # 4. Add heuristic candidates
    x = _x
    c = _c
    h_tri = np.maximum(1.0 - np.abs(x - c) / c, 0.0)
    candidates.append(h_tri)
    h_rc = 0.5 * (1.0 + np.cos(np.pi * (x - c) / c))
    candidates.append(h_rc)
    h_2b = np.exp(-((x - (c-5))**2)/5) + np.exp(-((x - (c+5))**2)/5)
    candidates.append(h_2b)
    d = np.abs(x - c)
    h_ham = 0.54 - 0.46 * np.cos(np.pi * d / c)
    h_ham = np.maximum(h_ham, 0.0)
    candidates.append(h_ham)
    h_blk = 0.42 - 0.5 * np.cos(np.pi * d / c) + 0.08 * np.cos(2.0 * np.pi * d / c)
    h_blk = np.maximum(h_blk, 0.0)
    candidates.append(h_blk)
    
    # 5. Local Refinement using L-BFGS-B in log-space with annealing
    for h_cand in candidates:
        if time.time() > deadline - 15.0:
            break
            
        # Ensure valid start
        h_cand = np.maximum(h_cand, 1e-6)
        
        # Log space start
        log_h_start = np.log(h_cand)
        
        try:
            # Stage 1: Smooth with low beta
            res1 = minimize(
                neg_value_smooth,
                log_h_start,
                args=(5.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_stage1 = np.exp(res1.x - np.mean(res1.x))
            h_stage1 = np.maximum(h_stage1, 1e-8)
            
            # Stage 2: Smooth with higher beta
            log_h_s2 = np.log(h_stage1)
            res2 = minimize(
                neg_value_smooth,
                log_h_s2,
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_stage2 = np.exp(res2.x - np.mean(res2.x))
            h_stage2 = np.maximum(h_stage2, 1e-8)
            
            # Stage 3: Exact objective
            log_h_s3 = np.log(h_stage2)
            res3 = minimize(
                neg_value,
                log_h_s3,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_refined = np.exp(res3.x - np.mean(res3.x))
            h_refined = np.maximum(h_refined, 0.0)
            val_refined = compute_value(h_refined)
            
            if val_refined > best_val:
                best_val = val_refined
                best_h = h_refined.copy()
                
        except Exception:
            pass
    
    # 6. Final Polish with annealing on the absolute best
    if time.time() < deadline - 5.0:
        log_h_best = np.log(np.maximum(best_h, 1e-6))
        try:
            # Anneal beta from 10 to 500
            betas = [10.0, 50.0, 200.0, 500.0]
            current_log_h = log_h_best.copy()
            for beta in betas:
                if time.time() > deadline - 3.0:
                    break
                res = minimize(
                    neg_value_smooth,
                    current_log_h,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                current_log_h = res.x
            
            h_annealed = np.exp(current_log_h - np.mean(current_log_h))
            h_annealed = np.maximum(h_annealed, 0.0)
            
            # Final exact polish
            log_h_final = np.log(np.maximum(h_annealed, 1e-6))
            res_final = minimize(
                neg_value,
                log_h_final,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            h_final = np.exp(res_final.x - np.mean(res_final.x))
            h_final = np.maximum(h_final, 0.0)
            val_final = compute_value(h_final)
            
            if val_final > best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass
            
    # Ensure output validity
    best_h = np.asarray(best_h, dtype=float)
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 1.0)
    
    return list(best_h)

# Test block (optional, remove for submission if not needed, but good for sanity)
if __name__ == "__main__":
    h = construct()
    val = compute_value(h)
    print(f"Value: {val}")
    print(f"Height sample: {h[:5]}")
