"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 600
DEADLINE = 145.0  # safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Uniform baseline
    x_uniform = np.ones(N)
    
    # Compute the objective function
    def objective(x):
        # Autoconvolution
        conv = fftconvolve(x, x, mode='full')
        # Normalize: divide by sum of x squared (or similar normalization)
        # The "normalized maximum of autoconvolution" - let's use max(conv) / sum(x)^2
        # or max(conv) / (sum(x * x))
        # Let's try max(conv) / sum(x)^2 first
        norm = np.sum(x) ** 2
        if norm < 1e-10:
            return 1e10
        return np.max(conv) / norm
    
    # Initial uniform value
    best_x = x_uniform.copy()
    best_val = objective(best_x)
    
    # Try optimization with L-BFGS-B
    try:
        # Use a smooth surrogate: soft-max via log-sum-exp
        def soft_max(x, beta=10.0):
            conv = fftconvolve(x, x, mode='full')
            norm = np.sum(x) ** 2
            if norm < 1e-10:
                return 1e10
            conv_norm = conv / norm
            # Log-sum-exp for numerical stability
            m = np.max(conv_norm)
            return m + np.log(np.sum(np.exp(conv_norm - m))) / len(conv_norm)
        
        # Use the soft-max as a smooth surrogate for the max
        def obj_smooth(x, beta=10.0):
            conv = fftconvolve(x, x, mode='full')
            norm = np.sum(x) ** 2
            if norm < 1e-10:
                return 1e10
            conv_norm = conv / norm
            # Log-sum-exp approximation of max
            m = np.max(conv_norm)
            return m + np.log(np.sum(np.exp(conv_norm - m)) / len(conv_norm))
        
        # Try multiple initializations
        inits = [
            np.ones(N),
            np.concatenate([np.ones(N//2), np.zeros(N//2)]),
            np.random.dirichlet(np.ones(N) * 10) * N,  # random Dirichlet
        ]
        
        for init in inits:
            if time.time() > DEADLINE - 5:
                break
            
            x0 = init.copy()
            # Ensure nonnegative
            x0 = np.maximum(x0, 0)
            
            res = minimize(
                obj_smooth,
                x0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_val:
                x_opt = np.maximum(res.x, 0)
                val = objective(x_opt)
                if val < best_val:
                    best_val = val
                    best_x = x_opt.copy()
            
            if time.time() > DEADLINE - 10:
                break
        
        # Final polish with tighter tolerance on the true objective
        if time.time() < DEADLINE - 5:
            res2 = minimize(
                lambda x: objective(np.maximum(x, 0)),
                best_x,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res2.fun < best_val:
                x_opt = np.maximum(res2.x, 0)
                val = objective(x_opt)
                if val < best_val:
                    best_val = val
                    best_x = x_opt.copy()
    
    except Exception:
        pass
    
    return list(best_x)
# EVOLVE-BLOCK-END
