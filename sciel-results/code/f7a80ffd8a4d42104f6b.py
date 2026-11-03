import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140

def compute_score(f):
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def cosine_basis(x, centers, widths, heights):
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components):
    f = parametric_to_step(params, n_components)
    return compute_score(f)

def local_refinement(h, start_time, max_time, alpha_init=0.05):
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12: break
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            
        threshold = max_c * 0.95
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12: break
        h_new = h_new * (N / s_new)
        
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.3)
        else:
            alpha *= 0.9
            if alpha < 1e-6: break
            
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Start with constant
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try 3 symmetric bumps
    n_c = 3
    # Initial guess: centers at -0.1, 0, 0.1, widths 0.2, heights 1
    x0 = np.array([-0.1, 0.0, 0.1, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0])
    
    if time.time() - start_time < TIME_LIMIT - 20:
        res = minimize(objective, x0, args=(n_c,), method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.success and res.fun < best_score:
            f_opt = parametric_to_step(res.x, n_c)
            best_f, best_score = local_refinement(f_opt, time.time(), TIME_LIMIT - 5)
            
    # Try 1 bump (cosine)
    n_c = 1
    x0 = np.array([0.0, 0.4, 1.0])
    if time.time() - start_time < TIME_LIMIT - 10:
        res = minimize(objective, x0, args=(n_c,), method='Nelder-Mead', 
                       options={'maxiter': 200})
        if res.success and res.fun < best_score:
            f_opt = parametric_to_step(res.x, n_c)
            best_f, best_score = local_refinement(f_opt, time.time(), TIME_LIMIT - 2)

    return list(best_f)
