import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Objective function
    def objective(h):
        # h is flattened
        h = h.reshape(N)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        conv = np.convolve(h, h)
        # Smooth max
        beta = 50.0
        max_val = (1.0 / beta) * np.log(np.sum(np.exp(beta * conv)))
        return max_val / (s * s)
        
    def gradient(h):
        # Numerical gradient or analytic? 
        # Nelder-Mead doesn't need it.
        return None

    best_h = None
    best_val = np.inf
    
    # Multi-start
    # Start 1: Random
    # Start 2: Sine wave
    # Start 3: Linear ramp
    
    for i in range(10):
        if time.time() > deadline:
            break
            
        if i == 0:
            h0 = np.random.randn(N)
        elif i == 1:
            h0 = np.sin(np.linspace(0, 10, N))
        elif i == 2:
            h0 = np.linspace(-1, 1, N)
        else:
            h0 = np.random.randn(N)
            
        # Normalize to avoid scaling issues during optimization
        h0 = h0 / np.sum(h0) if np.sum(h0) != 0 else h0
        
        try:
            res = minimize(objective, h0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x
        except Exception:
            pass

    if best_h is None:
        return np.ones(N)
        
    return best_h
