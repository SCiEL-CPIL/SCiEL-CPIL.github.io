"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Width of each interval
    w = 1.0 / 300.0  # total length is 1/2, so 600 intervals of width 1/300
    
    def autoconv_max(f):
        """Compute max of autoconvolution of step function f."""
        # Pad for linear convolution
        n_fft = 2 * N
        F = np.fft.rfft(f, n=n_fft)
        ac = np.fft.irfft(F * F.conj(), n=n_fft)
        # ac[0] is the zero-lag, ac[1:N] are positive lags
        # For a step function, the continuous autoconvolution at lag t is
        # integral f(x)f(x+t) dx. For step functions, this is piecewise.
        # The peak is at lag 0: sum f_i^2 * w
        # But we want the max over all lags
        # The discrete autoconvolution (unnormalized) at lag k is:
        # sum_i f[i]*f[i+k] for valid i
        # The continuous version multiplies by w
        # Max is at k=0: sum f[i]^2 * w
        # But wait, for nonnegative f, the autoconvolution is maximized at lag 0
        # So max_autoconv = w * sum(f^2)
        # The "normalized" version - let's assume it's max_autoconv / (integral f)^2
        # integral f = sum(f) * w
        # So normalized = (w * sum(f^2)) / (w^2 * sum(f)^2) = sum(f^2) / (w * sum(f)^2)
        # Or maybe it's just the peak value of the continuous autoconvolution
        # Let me try: score = max of continuous autoconv / (L1 norm)^2
        # L1 norm = sum(f) * w
        # max autoconv = w * sum(f^2)
        # score = w * sum(f^2) / (w * sum(f))^2 = sum(f^2) / (w * sum(f)^2)
        
        s = np.sum(f)
        if s < 1e-10:
            return 1e10
        sq = np.sum(f * f)
        return sq / (w * s * s)
    
    def objective(x):
        f = np.abs(x)  # Enforce nonnegativity
        return autoconv_max(f)
    
    # Start with uniform distribution
    x0 = np.ones(N)
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try a few optimization strategies with time budget
    methods_to_try = [
        {'method': 'L-BFGS-B', 'options': {'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}},
        {'method': 'Nelder-Mead', 'options': {'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12}},
    ]
    
    for method_info in methods_to_try:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(objective, best_x, method=method_info['method'],
                            options=method_info['options'],
                            bounds=[(0, None)] * N)
            if result.fun < best_val:
                best_val = result.fun
                best_x = np.abs(result.x)
        except Exception:
            pass
    
    # If time remains, try a coordinate descent refinement
    if time.time() < deadline - 5:
        f = best_x.copy()
        for iteration in range(50000):
            if time.time() > deadline - 2:
                break
            # Random coordinate descent
            i = np.random.randint(N)
            old_val = f[i]
            # Try small perturbations
            for delta in [-0.1, -0.01, 0.01, 0.1]:
                new_val = max(0, old_val + delta)
                f[i] = new_val
                new_obj = objective(f)
                if new_obj < best_val:
                    best_val = new_obj
                    best_x = f.copy()
                else:
                    f[i] = old_val
                    break
    
    return list(np.abs(best_x))
# EVOLVE-BLOCK-END
