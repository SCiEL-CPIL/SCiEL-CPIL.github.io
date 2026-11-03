"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: 
1. Initialize with diverse spectral/time-domain candidates.
2. Refine using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) 
   with analytic gradients of the smoothed autoconvolution objective.
3. Enforce non-negativity via projection and normalization.
4. Final polish with L-BFGS-B if time permits.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def compute_grad(f, beta=20.0):
    """
    Compute the gradient of the smoothed objective J(f) = smooth_max(f*f) / (sum(f))^2.
    Uses log-sum-exp for the smooth max surrogate.
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.sum(f)
    if s < 1e-15:
        return np.zeros_like(f)
    
    conv = np.convolve(f, f, mode='full')
    conv_max = np.max(conv)
    if conv_max < 1e-15:
        return np.zeros_like(f)
    
    # Softmax weights for smoothed max
    shifted = conv - conv_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    w = exp_shifted / sum_exp
    
    # Smooth max value K
    K = conv_max + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of K w.r.t f: d(conv[m])/d(f[j]) = f[m-j] (for valid m-j)
    # grad_K[j] = 2 * sum_m w[m] * f[m-j]
    # This is a cross-correlation: grad_K = 2 * correlate(w, f) truncated to N
    grad_K = 2.0 * np.convolve(w, f, mode='full')[:N]
    
    # J = K / s^2
    # dJ/d(f[j]) = grad_K[j]/s^2 - 2*K/(s^3) * 1
    grad_J = grad_K / (s * s) - (2.0 * K / (s * s * s)) * np.ones(N)
    
    return grad_J

def smooth_max_and_grad(f, beta=20.0):
    """
    Compute the smooth max of autoconvolution and its gradient.
    Returns (value, gradient).
    """
    s = np.sum(f)
    if s < 1e-15:
        return 0.0, np.zeros_like(f)
    
    conv = np.convolve(f, f, mode='full')
    conv_max = np.max(conv)
    if conv_max < 1e-15:
        return 0.0, np.zeros_like(f)
    
    # Softmax weights for smoothed max
    shifted = conv - conv_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    w = exp_shifted / sum_exp
    
    # Smooth max value K
    K = conv_max + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of K w.r.t f
    # d(conv[m])/d(f[j]) = f[m-j]
    # grad_K[j] = 2 * sum_m w[m] * f[m-j]
    grad_K = 2.0 * np.convolve(w, f, mode='full')[:N]
    
    return K, grad_K

def fista_optimize(f_init, max_iter=2000, beta=20.0, time_limit=None, start_time=0):
    """
    FISTA optimization for minimizing smooth_max(f*f) subject to f >= 0 and sum(f) = N.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-12:
        return f
    f = f * (N / s)
    
    y = f.copy()
    t = 1.0
    L = 5.0 * np.max(f) + 1e-6
    
    for k in range(max_iter):
        if time_limit and time.time() - start_time > time_limit:
            break
        
        # Gradient step
        K_val, grad_K = smooth_max_and_grad(y, beta=beta)
        
        # Step: z = y - (1/L) * grad_K
        z = y - (1.0 / L) * grad_K
        
        # Project onto constraint set: sum(z) = N, z >= 0
        v = np.maximum(z, 0.0)
        s_v = np.sum(v)
        if s_v < 1e-12:
            v = np.ones(N) / N
        else:
            v = v * (N / s_v)
        
        # Update y for next iteration (Nesterov momentum)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = v + ((t - 1.0) / t_new) * (v - f)
        t = t_new
        f = v
        
        if k % 50 == 0:
            L = 5.0 * np.max(f) + 1e-6
            
    return f

def construct():
    start_time = time.time()
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    rng = np.random.RandomState(42)
    
    candidates = []
    
    # 1. Gaussian
    f0 = np.exp(-10 * x**2)
    f0 = f0 / np.sum(f0)
    candidates.append(f0)
    
    # 2. Broad Gaussian
    f1 = np.exp(-2 * x**2)
    f1 = f1 / np.sum(f1)
    candidates.append(f1)
    
    # 3. Uniform with small noise
    f2 = np.ones(N) * (1.0 + 0.1 * rng.randn(N))
    f2 = np.maximum(f2, 0.0)
    f2 = f2 / np.sum(f2)
    candidates.append(f2)
    
    # 4. Cosine bump
    f3 = np.cos(np.pi * x / 0.5) ** 2
    f3 = np.maximum(f3, 0.0)
    f3 = f3 / np.sum(f3)
    candidates.append(f3)
    
    # 5. Spectral init: low freq dominant
    F_init = np.zeros(N // 2 + 1)
    F_init[0] = 1.0
    F_init[1] = 0.5
    F_init[2] = 0.2
    f4 = np.fft.irfft(F_init, n=N)
    f4 = np.maximum(f4, 0.0)
    f4 = f4 / np.sum(f4)
    candidates.append(f4)
    
    for idx, f_init in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Run FISTA
        f_res = fista_optimize(f_init, max_iter=1000, beta=20.0, 
                               time_limit=TIME_LIMIT - 10, start_time=start_time)
        
        f_res = np.maximum(f_res, 0.0)
        s_res = np.sum(f_res)
        if s_res > 1e-12:
            f_res = f_res / s_res
            score_res = compute_score(f_res)
            if score_res < best_score:
                best_score = score_res
                best_f = f_res.copy()
        
        # Quick local refinement with L-BFGS-B on the best candidate found so far
        # Only if we have time
        if time.time() - start_time < TIME_LIMIT - 30:
             def objective_and_grad(f, _beta=15.0):
                f_clipped = np.maximum(f, 0.0)
                s = np.sum(f_clipped)
                if s < 1e-15:
                    return 1e10, np.zeros_like(f)
                conv = np.convolve(f_clipped, f_clipped, mode='full')
                val = np.max(conv) / (s * s)
                grad = compute_grad(f_clipped, beta=_beta)
                return val, grad
            
             try:
                result = minimize(
                    objective_and_grad, best_f, jac=True,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-5, 'maxls': 10}
                )
                f_res2 = np.maximum(result.x, 0.0)
                s_res2 = np.sum(f_res2)
                if s_res2 > 1e-12:
                    f_res2 = f_res2 / s_res2
                    score_res2 = compute_score(f_res2)
                    if score_res2 < best_score:
                        best_score = score_res2
                        best_f = f_res2.copy()
             except Exception:
                pass

    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
