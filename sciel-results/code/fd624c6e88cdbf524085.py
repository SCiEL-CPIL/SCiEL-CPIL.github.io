import numpy as np
import time
from scipy.optimize import minimize

N = 600
FFT_SIZE = 1 << (2*N - 1).bit_length()  # Precompute FFT size

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for 150s limit
    
    def compute_autoconv_peak(f):
        """Compute max of linear autoconvolution of f."""
        F = np.fft.rfft(f, n=FFT_SIZE)
        G = np.fft.irfft(F * F, n=FFT_SIZE)
        return np.max(G)
    
    def true_metric(f):
        """Compute max(f*f) / sum(f^2)."""
        l2_sq = np.sum(f * f)
        if l2_sq < 1e-15:
            return 1e12
        peak = compute_autoconv_peak(f)
        return peak / l2_sq
    
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    def l2_normalize(f):
        norm = np.sqrt(np.sum(f * f))
        if norm < 1e-15:
            return f
        return f / norm
    
    # Beta for smooth max approximation
    beta = 200.0
    
    def smooth_obj_l2(y):
        """Minimize smooth max(f*f) / sum(f^2) using L2 normalization."""
        f = softmax(y)
        f = l2_normalize(f)
        F = np.fft.rfft(f, n=FFT_SIZE)
        G = np.fft.irfft(F * F, n=FFT_SIZE)
        # Smooth max of G
        max_G = np.log(np.sum(np.exp(beta * G))) / beta
        # Since f is L2-normalized, sum(f^2) = 1, so metric = max_G
        return max_G
    
    # Generate candidate initializations
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N) / np.sqrt(N))
    
    # 2. Gaussian (narrow)
    t = np.arange(N) - (N-1)/2.0
    for sigma_frac in [0.1, 0.2, 0.3, 0.5]:
        sigma = sigma_frac * (N-1)
        gauss = np.exp(-t**2 / (2*sigma**2))
        gauss = l2_normalize(gauss)
        candidates.append(gauss)
    
    # 3. Hann window
    x = np.linspace(0, 1, N)
    hann = np.sin(np.pi * x)**2
    hann = l2_normalize(hann)
    candidates.append(hann)
    
    # 4. Triangle
    tri = np.minimum(x, 1-x) * 2
    tri = l2_normalize(tri)
    candidates.append(tri)
    
    # Evaluate all candidates
    best_f = candidates[0]
    best_val = true_metric(best_f)
    
    for f in candidates[1:]:
        val = true_metric(f)
        if val < best_val:
            best_val = val
            best_f = f
    
    # Optimize from each of the top candidates
    top_candidates = candidates[:5]  # Optimize from best few
    
    for init_f in top_candidates:
        if time.time() > deadline - 10:
            break
        
        # Convert to y-space: y = log(f) works as initial for softmax
        y_init = np.log(init_f + 1e-15)
        
        try:
            res = minimize(smooth_obj_l2, y_init, method='L-BFGS-B', 
                          options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if hasattr(res, 'x'):
                f_opt = softmax(res.x)
                f_opt = l2_normalize(f_opt)
                val_opt = true_metric(f_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_f = f_opt
        except Exception:
            pass
    
    # Final polish: try a few more iterations from the best
    if time.time() < deadline - 5:
        y_init = np.log(best_f + 1e-15)
        try:
            res = minimize(smooth_obj_l2, y_init, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            if hasattr(res, 'x'):
                f_polish = softmax(res.x)
                f_polish = l2_normalize(f_polish)
                val_polish = true_metric(f_polish)
                if val_polish < best_val:
                    best_val = val_polish
                    best_f = f_polish
        except Exception:
            pass
    
    return list(best_f)
