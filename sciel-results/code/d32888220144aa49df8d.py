import numpy as np
from scipy.optimize import minimize

N = 50

def compute_value_from_x(x):
    # Softmax parameterization to ensure h >= 0 and sum(h) = 1
    x_max = np.max(x)
    h = np.exp(x - x_max)
    h_sum = np.sum(h)
    if h_sum == 0:
        return 0.0
    h = h / h_sum
    
    g = np.convolve(h, h, mode='full')
    
    # Simplified value formula derived from task definition:
    # value = (2 * sum(g^2) + sum(g[:-1]*g[1:])) / (3 * max(g))
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:])
    max_g = np.max(g)
    
    if max_g == 0:
        return 0.0
        
    return (2.0 * g_sq_sum + cross_sum) / (3.0 * max_g)

def construct():
    # Initial guesses in x-space (log-space of h)
    # Uniform h corresponds to x=0
    x_uniform = np.zeros(N)
    
    # Peaked h corresponds to peaked x
    x_peaked = -np.abs(np.linspace(-1, 1, N)) * 5
    
    # Random
    np.random.seed(42)
    x_random = np.random.randn(N)
    
    # Another shape
    x_gauss = -np.linspace(-1, 1, N)**2 * 10
    
    seeds = [x_uniform, x_peaked, x_random, x_gauss]
    
    best_val = -np.inf
    best_h = None
    
    for x_start in seeds:
        def objective(x):
            val = compute_value_from_x(x)
            return -val
        
        # Use L-BFGS-B for gradient-based optimization
        res = minimize(objective, x_start, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14})
        
        # Recover h
        x_opt = res.x
        x_max = np.max(x_opt)
        h_opt = np.exp(x_opt - x_max)
        h_opt = h_opt / np.sum(h_opt)
        
        val_opt = compute_value_from_x(x_opt)
        
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt
            
    return list(best_h)
