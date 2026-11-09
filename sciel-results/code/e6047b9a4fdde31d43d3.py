import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator: Duffing-like core + sparse RBF corrections.
    
    Core: a = -x*(c0 + c1*x^2) - v*(c2 + c3*x) + c4 + c5*x + c6*v + c7*v^2
    RBF:  sum_k w_k * exp(-d_k^2 / (2*sigma^2))
    
    7 core params + 3 RBF weights = 10 total.
    RBF centers fixed at strategic points in (X, V) space.
    """
    t0 = time.time()
    
    # Scale inputs
    X = x / 0.55
    V = v / 0.25
    
    p0, p1, p2, p3, p4, p5, p6, p7, w1, w2, w3 = params[:10]
    
    # Duffing-like core with quadratic terms
    core = (-X * (p0 + p1 * X**2) 
            - V * (p2 + p3 * X) 
            + p4 
            + p5 * X 
            + p6 * V 
            + p7 * V**2)
    
    # Fixed RBF centers in scaled (X, V) space
    # Centers chosen to cover the domain well
    cx1, cv1 = 0.0, 0.0      # center
    cx2, cv2 = 0.7, 0.5      # upper-right
    cx3, cv3 = -0.7, -0.5    # lower-left
    
    sigma2 = 2 * (0.4**2)  # bandwidth
    
    rbf1 = w1 * np.exp(-((X - cx1)**2 + (V - cv1)**2) / sigma2)
    rbf2 = w2 * np.exp(-((X - cx2)**2 + (V - cv2)**2) / sigma2)
    rbf3 = w3 * np.exp(-((X - cx3)**2 + (V - cv3)**2) / sigma2)
    
    result = core + rbf1 + rbf2 + rbf3
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
