import numpy as np
from scipy.optimize import minimize

def compute_value(h):
    h = np.array(h)
    g = np.convolve(h, h)
    m = len(g)
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # spacing w
    # The points are np.linspace(-0.5, 0.5, m + 2)
    # Number of intervals is m + 1
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # Note: i goes from 0 to m. y has indices 0 to m+1.
    # So we use y[0:m+1] and y[1:m+2]
    term1 = y[0:m+1]**2
    term2 = y[0:m+1] * y[1:m+2]
    term3 = y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(term1 + term2 + term3)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def neg_value(h):
    return -compute_value(h)

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use multiple random restarts with Powell's method
    # Powell's method is robust for non-smooth functions and box constraints
    best_h = None
    best_val = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several different initial guesses
    # 1. Gaussian centered
    # 2. Uniform
    # 3. Random positive
    # 4. Triangle
    
    c = N // 2
    x = np.arange(N)
    
    initial_guesses = [
        np.exp(-0.1 * (x - c)**2),  # Gaussian
        np.ones(N),                  # Uniform
        rng.random(N),               # Random
        np.maximum(0, 1 - np.abs(x - c) / (N/2)), # Triangle
        np.exp(-0.05 * (x - c)**2),  # Wider Gaussian
        np.exp(-0.2 * (x - c)**2)    # Narrower Gaussian
    ]
    
    bounds = [(0, 10.0)] * N
    
    for h0 in initial_guesses:
        # Scale initial guess to be in a reasonable range
        if np.max(h0) > 0:
            h0 = h0 / np.max(h0) * 5.0 # Scale to max 5
        
        try:
            res = minimize(neg_value, h0, method='Powell', 
                           bounds=bounds,
                           options={'maxiter': 500, 'maxfev': 10000, 'ftol': 1e-8, 'xtol': 1e-8})
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            
            val = compute_value(h_opt)
            
            if val > best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            continue
            
    if best_h is None:
        # Fallback to seed
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
