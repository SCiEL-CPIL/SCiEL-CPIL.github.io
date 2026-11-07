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
    # Initial guess: Gaussian-like
    c = N // 2
    x = np.arange(N)
    h0 = np.exp(-0.1 * (x - c)**2)
    
    # Define bounds
    bounds = [(0, 10.0)] * N
    
    # Optimize
    # Use Nelder-Mead, robust for non-smooth (due to max in norm_inf)
    res = minimize(neg_value, h0, method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
    
    h_opt = res.x
    # Ensure non-negative
    h_opt = np.maximum(h_opt, 0)
    
    return h_opt
# EVOLVE-BLOCK-END
