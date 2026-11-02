import numpy as np
from scipy.optimize import minimize
import time

N = 600
N_FFT = 2048

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    # Initial guess: Hanning window
    f0 = np.hanning(N)
    # Ensure non-negative (hanning is non-negative)
    f0 = np.abs(f0)
    # Normalize
    norm = np.linalg.norm(f0)
    if norm > 0:
        f0 = f0 / norm
    
    def objective(f):
        # f is flattened
        f = np.abs(f) # Enforce non-negativity in objective? 
        # Actually, L-BFGS-B handles bounds, so f should be >= 0.
        # But numerical errors might make it slightly negative.
        
        # Normalize
        norm = np.linalg.norm(f)
        if norm < 1e-10:
            return 1e6
        f_norm = f / norm
        
        # Compute autocorrelation
        # Zero pad
        fp = np.zeros(N_FFT)
        fp[:N] = f_norm
        
        # FFT
        F = np.fft.fft(fp)
        # Power spectrum
        P = np.abs(F)**2
        # IFFT to get autocorrelation
        A = np.real(np.fft.ifft(P))
        
        # A[0] is the peak (lag 0)
        # Sidelobes are A[1:N] and A[N_FFT-N+1:N_FFT] (negative lags)
        # Due to symmetry, A[k] == A[-k].
        # So we just need max of A[1:N]
        
        # Clip small negative values due to numerical noise
        sidelobes = A[1:N]
        sidelobes = np.maximum(sidelobes, 0)
        
        # Soft max
        beta = 50.0
        max_val = np.max(sidelobes)
        if max_val == 0:
            return 0.0
        
        # Log-sum-exp for stability
        # sum(exp(beta * (s - max_val)))
        shifted = beta * (sidelobes - max_val)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_max = max_val + (1.0/beta) * log_sum
        
        return soft_max
    
    # Bounds
    bounds = [(0, None) for _ in range(N)]
    
    # Optimization
    best_f = f0
    best_val = objective(f0)
    
    try:
        res = minimize(
            objective, 
            f0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
            pass # Just break out if we were in a loop, but minimize is a single call.
            
        if res.fun < best_val:
            best_f = res.x
            best_val = res.fun
            
    except Exception as e:
        # If optimization fails, return the initial guess
        pass
    
    # Final normalization and cleanup
    best_f = np.abs(best_f)
    norm = np.linalg.norm(best_f)
    if norm > 0:
        best_f = best_f / norm
        
    return list(best_f)
