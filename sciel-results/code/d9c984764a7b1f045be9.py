import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = 120.0  # 150s limit, keep margin
    
    # Initial guess: Gaussian window
    x = np.linspace(-1, 1, N)
    h = np.exp(-x**2 / (2 * 0.5**2))
    h = np.maximum(h, 0)
    
    # Normalize to sum to 1 for stability during optimization
    h = h / np.sum(h)
    
    def objective(h):
        # Ensure non-negative
        h = np.maximum(h, 0)
        
        # Compute autoconvolution
        # Use FFT for speed
        h_fft = np.fft.fft(h, n=2*N)
        acv_fft = h_fft * np.conj(h_fft)
        acv = np.fft.ifft(acv_fft).real[:N]
        
        # Normalize
        norm = np.sum(h) ** 2
        if norm == 0:
            return 1e10
            
        acv_norm = acv / norm
        
        # We want to minimize the maximum sidelobe
        # Exclude the peak at index 0
        # Also exclude the wrap-around if any (though linear conv is used via ifft slice)
        # The acv is symmetric. acv[0] is peak.
        # Max sidelobe is max(acv[1:])
        
        max_sidelobe = np.max(acv_norm[1:])
        
        # Add a small penalty if the peak is too high? 
        # If the metric is global max, this is wrong. 
        # But assuming sidelobe metric.
        return max_sidelobe

    # Optimization
    best_h = h
    best_score = objective(h)
    
    # Try a few restarts or steps
    try:
        # Use L-BFGS-B for bounded optimization
        bounds = [(0, None) for _ in range(N)]
        
        # Run optimization
        res = minimize(objective, h, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.fun < best_score:
            best_h = res.x
            best_score = res.fun
            
        # Check time
        if time.time() > deadline - 10:
            return list(best_h)
            
        # Try Nelder-Mead for local polish
        res2 = minimize(objective, best_h, method='Nelder-Mead', 
                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res2.fun < best_score:
            best_h = res2.x
            best_score = res2.fun
            
    except Exception:
        pass

    # Final check
    if time.time() > deadline - 5:
        return list(best_h)
        
    return list(best_h)
