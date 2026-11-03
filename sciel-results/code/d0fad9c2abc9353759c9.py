import numpy as np
from scipy.optimize import minimize
import time

N = 600

def compute_score(h):
    # h is array of heights
    if np.any(h < 0):
        return np.inf
    S = np.sum(h)
    if S == 0:
        return np.inf
    # Linear convolution
    c = np.convolve(h, h)
    max_c = np.max(c)
    return max_c / (S * S)

def construct():
    start_time = time.time()
    deadline = 100.0 # 100s limit
    
    # Initial guess: Uniform
    h0 = np.ones(N)
    best_h = h0.copy()
    best_score = compute_score(best_h)
    
    # Try a few random symmetric perturbations
    # Symmetric shapes are often optimal for such norms
    
    def symmetric_h(params):
        # params define the first half, mirror to second half
        # Simple linear interpolation or just direct values
        # To keep it simple, use a parametric form: 
        # h_i = base + amp * cos(pi * i / N) ?
        # Or just optimize the first N/2 and mirror.
        
        pass

    # Given the time limit and complexity, a full 600D optimization is hard.
    # We use a parametric family: Super-ellipse or Cosine bump.
    # h_i = (1 - (2*i/N - 1)**p)**q
    
    def make_superellipse(p, q, base=0.0):
        x = np.linspace(-1, 1, N)
        h = np.maximum(0, 1 - np.abs(x)**p)**q
        h = h + base
        return h

    # Grid search over p, q
    best_p = 2.0
    best_q = 1.0
    
    p_range = np.linspace(0.5, 4.0, 10)
    q_range = np.linspace(0.5, 4.0, 10)
    
    for p in p_range:
        if time.time() - start_time > deadline:
            break
        for q in q_range:
            if time.time() - start_time > deadline:
                break
            h = make_superellipse(p, q)
            s = compute_score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
                best_p = p
                best_q = q

    # Local refinement using Nelder-Mead on p, q
    if time.time() - start_time < deadline:
        def objective(params):
            p, q = params
            if p <= 0 or q <= 0:
                return 1e6
            h = make_superellipse(p, q)
            return compute_score(h)
        
        x0 = [best_p, best_q]
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_h = make_superellipse(res.x[0], res.x[1])
            
    # Final check and return
    # Ensure non-negative
    best_h = np.maximum(0, best_h)
    
    # If uniform is best, return uniform
    # The prompt asks for sequence of heights
    return list(best_h)
