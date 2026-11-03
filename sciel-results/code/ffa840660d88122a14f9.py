import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 140

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def project_simplex(v, s=1.0):
    """Project v onto the simplex {x | x >= 0, sum(x) = s}."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def smooth_max_objective(f, beta):
    """
    Compute the smooth approximation of the normalized max autoconvolution
    using the log-sum-exp trick.
    Returns the objective value and the gradient with respect to f.
    """
    f = np.asarray(f, dtype=np.float64)
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(f)
    
    conv = fft_autoconv(f)
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0, np.zeros_like(f)
    
    # Normalize f to sum=1 for the objective calculation to keep it scale invariant
    # But we want the gradient of max(conv) / s^2.
    # Let's just minimize max(conv) subject to sum(f)=1, f>=0.
    # The score is max(conv) since s=1.
    
    # Smooth max: LSE(conv) / beta
    # To avoid overflow:
    z = beta * (conv - max_conv)
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z)
    lse = max_conv + np.log(sum_exp_z) / beta
    
    # Gradient of LSE wrt f
    # p_k = exp(z_k) / sum(exp_z)
    p = exp_z / sum_exp_z
    
    # Gradient of conv_k wrt f_i
    # conv_k = sum_j f_j f_{j+k}
    # d conv_k / d f_i = f_{i+k} + f_{i-k} (with bounds)
    
    # We can compute the gradient vector g_i = sum_k p_k * (d conv_k / d f_i)
    # g_i = sum_k p_k * (f_{i+k} 1_{i+k<N} + f_{i-k} 1_{i-k>=0})
    
    # This is a correlation-like operation.
    # Let's compute it efficiently.
    # Note that p is defined for k in [0, 2N-2].
    # However, due to symmetry of autoconvolution, p_k = p_{2N-2-k} is not necessarily true 
    # unless the signal is symmetric. But conv is symmetric: conv_k = conv_{2N-2-k}.
    # Wait, linear autoconvolution is symmetric around (N-1).
    # conv[m] = sum_j f_j f_{j+m}
    # conv[2N-2-m] = sum_j f_j f_{j+2N-2-m}
    # Let j' = N-1-j.
    # This symmetry holds. So p is symmetric.
    
    # We can exploit this or just compute directly.
    # Direct computation O(N^2) is fine for N=600.
    # g_i = sum_{k=0}^{2N-2} p_k (f_{i+k} + f_{i-k})
    
    # Optimization:
    # g = np.zeros(N)
    # for i in range(N):
    #    # Term 1: f_{i+k} exists for k in [0, N-1-i]
    #    g[i] += np.dot(p[0:N-i], f[i:N])
    #    # Term 2: f_{i-k} exists for k in [0, i]
    #    g[i] += np.dot(p[0:i+1], f[i::-1])
    
    # This loop is O(N^2). 600^2 = 360k operations.
    # 500 iterations * 360k = 180M operations.
    # In NumPy, a dot product of length ~300 is fast.
    # But doing it in a Python loop for 600 items might be slow.
    
    # Let's vectorize using FFT if possible?
    # g_i = sum_k p_k f_{i+k} + sum_k p_k f_{i-k}
    # First term: (p * f) where p is padded?
    # This is a cross-correlation.
    # np.correlate(p, f, mode='full')?
    # p has length 2N-1. f has length N.
    # np.correlate(p, f, 'valid')? No.
    
    # Let's stick to the loop but optimize it.
    # Or use matrix multiplication?
    # Construct a matrix M where M[i, k] = 1 if i+k < N else 0.
    # Then first term is M @ f? No, p is weighted.
    # g1 = M_p @ f where M_p[i, k] = p_k if i+k < N else 0.
    # This is a banded matrix.
    
    # Let's try a simple loop first, it's robust.
    g = np.zeros(N)
    for i in range(N):
        # p[0:N-i] corresponds to f[i:N]
        if N - i > 0:
            g[i] += np.dot(p[0:N-i], f[i:N])
        # p[0:i+1] corresponds to f[i::-1] (which is f[i], f[i-1], ..., f[0])
        if i >= 0:
            g[i] += np.dot(p[0:i+1], f[i::-1])
            
    # The objective is lse.
    # The gradient is g.
    # We are minimizing lse.
    return lse, g

def spectral_smooth_descent(f_init, start_time, max_time=120.0):
    """
    Minimize the smooth max autoconvolution using projected gradient descent.
    This is a structural change from coordinate descent to a global smooth method.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    lr = 0.01
    beta = 10.0
    
    for it in range(1000):
        if time.time() - start_time > max_time:
            break
        
        obj, grad = smooth_max_objective(f, beta)
        
        # Gradient step
        f_new = f - lr * grad
        
        # Project to simplex
        f_new = project_simplex(f_new, 1.0)
        
        val = compute_score(f_new)
        if val < best_val:
            best_val = val
            best_f = f_new.copy()
            f = f_new
        else:
            lr *= 0.5
            # Optional: increase beta to sharpen the max approximation if stuck?
            # beta *= 1.1

    return best_f

def exact_coordinate_descent(f_init, start_time, max_time=120.0):
    """
    Coordinate descent where each coordinate is optimized exactly.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    for sweep in range(100):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            
            old_val = f[i]
            S = np.sum(f)
            
            conv_current = fft_autoconv(f)
            
            def objective(v):
                if v < 0:
                    return np.inf
                S_new = S - old_val + v
                if S_new < 1e-15:
                    return np.inf
                m_start = max(0, i)
                m_end = min(2 * N - 2, i + N - 1)
                
                conv_vals = conv_current[m_start:m_end + 1].copy()
                
                js = np.arange(m_start, m_end + 1) - i
                
                old_contribs = np.zeros(len(js))
                mask_diag = (js == i)
                mask_offdiag = ~mask_diag
                old_contribs[mask_diag] = old_val * old_val
                old_contribs[mask_offdiag] = 2 * old_val * f[js[mask_offdiag]]
                
                conv_bg = conv_vals - old_contribs
                
                new_contribs = np.zeros(len(js))
                new_contribs[mask_diag] = v * v
                new_contribs[mask_offdiag] = 2 * v * f[js[mask_offdiag]]
                
                new_conv_vals = conv_bg + new_contribs
                
                max_conv = np.max(new_conv_vals)
                return max_conv / (S_new * S_new)
            
            v_upper = max(1.0, 2 * old_val + 0.1)
            
            try:
                res = opt.minimize_scalar(objective, bounds=(0, v_upper), method='bounded')
                v_opt = res.x
            except Exception:
                v_opt = old_val
            
            if v_opt != old_val:
                f[i] = v_opt
                improved = True
        
        s = np.sum(f)
        if s > 1e-15:
            f = f / s
        
        current_val = compute_score(f)
        if current_val < best_val - 1e-12:
            best_val = current_val
            best_f = f.copy()
        
        if not improved:
            break
            
    return best_f

def construct():
    start_time = time.time()
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        f = np.exp(-alpha * x ** 2)
        candidates.append(f / np.sum(f))
        
    # Triangle
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
            
    # Hann Window
    f = 0.5 * (1 - np.cos(2 * np.pi * (x + 0.25)))
    f = np.maximum(f, 0)
    if np.sum(f) > 1e-15:
        candidates.append(f / np.sum(f))
        
    # Blackman Window
    t = (x + 0.25) / 0.5
    t = np.clip(t, 0, 1)
    f = 0.42 - 0.5 * np.cos(2 * np.pi * t) + 0.08 * np.cos(4 * np.pi * t)
    f = np.maximum(f, 0)
    if np.sum(f) > 1e-15:
        candidates.append(f / np.sum(f))
        
    # Random candidates
    np.random.seed(42)
    for _ in range(5):
        f = np.random.rand(N)
        candidates.append(f / np.sum(f))
        
    # Score them
    scored = []
    for f in candidates:
        score = compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply exact coordinate descent to the best initial candidates
    for f_init, score_init in scored[:5]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        refined = exact_coordinate_descent(f_init, start_time, TIME_LIMIT - 2)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score
            
    # Try spectral smooth descent on the best result
    if time.time() - start_time < TIME_LIMIT - 10:
        spectral_refined = spectral_smooth_descent(best_f, start_time, TIME_LIMIT - 2)
        spectral_score = compute_score(spectral_refined)
        if spectral_score < best_score:
            best_f = spectral_refined
            best_score = spectral_score
            
    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
