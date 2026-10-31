import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    # Parameterize h_i as A_0 + sum_{k=1}^M A_k cos(2*pi*k*i/N)
    M = 10
    # Initial guess: Uniform distribution (A_0 = 1/N, others 0)
    A0 = 1.0 / N
    A_init = np.zeros(M + 1)
    A_init[0] = A0
    
    def coeffs_to_h(A):
        i = np.arange(N)
        h = np.zeros(N)
        for k in range(M + 1):
            if k == 0:
                h += A[0]
            else:
                h += A[k] * np.cos(2 * np.pi * k * i / N)
        return h

    def objective(A):
        h = coeffs_to_h(A)
        # Penalty for negative values
        neg = np.sum(np.maximum(0, -h)**2)
        if neg > 1e-6:
            return 1e6 + neg
        
        # Normalize to sum 1
        s = np.sum(h)
        if s <= 0:
            return 1e6
        h_norm = h / s
        
        # Compute autoconvolution (linear)
        g = np.convolve(h_norm, h_norm, mode='full')
        max_g = np.max(g)
        
        return max_g

    # Remez-like iterative minimax approach:
    # 1. Start with Uniform.
    # 2. Use a robust derivative-free optimizer (Nelder-Mead) to minimize the peak.
    # 3. If the result is worse than Uniform, revert to Uniform.
    
    # Uniform baseline
    h_uniform = np.ones(N) / N
    g_uniform = np.convolve(h_uniform, h_uniform, mode='full')
    peak_uniform = np.max(g_uniform)
    
    # Optimization
    # Using Nelder-Mead which is robust for non-smooth objectives like max
    res = minimize(objective, A_init, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
    
    A_opt = res.x
    h_opt = coeffs_to_h(A_opt)
    
    # Ensure non-negative and normalize
    h_opt = np.maximum(h_opt, 0)
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    else:
        h_opt = h_uniform
        
    g_opt = np.convolve(h_opt, h_opt, mode='full')
    peak_opt = np.max(g_opt)
    
    # Choose the better result
    if peak_opt < peak_uniform:
        return list(h_opt)
    else:
        return list(h_uniform)
