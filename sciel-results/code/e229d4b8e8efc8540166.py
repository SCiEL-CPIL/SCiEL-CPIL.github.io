import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    """
    Computes the value for the given step heights h.
    h: array of N nonnegative floats.
    """
    h = np.asarray(h)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
        
    g = np.convolve(h, h)
    m = len(g)
    
    # Construct y
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    # Spacing w
    # Points are np.linspace(-0.5, 0.5, m + 2)
    # Number of intervals is m + 1
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # i goes from 0 to m inclusive.
    # y indices involved: 0 to m+1.
    y_i = y[0:m+1]
    y_ip1 = y[1:m+2]
    
    terms = y_i**2 + y_i * y_ip1 + y_ip1**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def negative_value(z):
    """
    Objective function for optimization.
    z: unconstrained variables.
    h = z^2
    """
    h = z * z
    val = compute_value(h)
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Initial guess: small positive values to avoid zero gradients issues initially
    # We want to explore different shapes.
    # A good start might be a smooth bump or random.
    # Let's try a few random restarts or a smart initialization.
    
    best_h = None
    best_val = -np.inf
    
    # Try a few different initializations
    inits = []
    # 1. Constant
    inits.append(np.ones(N) * 0.1)
    # 2. Gaussian-like
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2) * 0.5)
    # 3. Random positive
    rng = np.random.RandomState(42)
    inits.append(rng.rand(N) * 0.5)
    # 4. Triangular
    inits.append(np.abs(1 - 2 * np.linspace(0, 1, N)) * 0.5)
    
    for z0 in inits:
        if time.time() > deadline:
            break
            
        # Optimize z
        try:
            res = minimize(negative_value, z0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            z_opt = res.x
            h_opt = z_opt * z_opt
            
            # Ensure non-negative and finite
            h_opt = np.maximum(h_opt, 0)
            
            val = compute_value(h_opt)
            
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception:
            continue
            
    if best_h is None:
        # Fallback to seed
        return [1.0] * N
        
    return list(best_h)
