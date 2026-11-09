import numpy as np
import time
import scipy.optimize as opt

N = 50

def compute_ratio(h):
    """
    Compute the ratio (L2^2) / SupConv.
    Assumes h is normalized to sum 1.
    """
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s == 0:
        return 0.0
    h_norm = h / s
    l2_sq = np.sum(h_norm**2)
    
    # Linear convolution
    conv = np.convolve(h_norm, h_norm, mode='full')
    sup_conv = np.max(conv)
    
    if sup_conv == 0:
        return 0.0
    ratio = l2_sq / sup_conv
    return ratio

def optimize_sparse(h0, indices, max_iter=1000):
    """
    Optimize the values at 'indices'.
    """
    if len(indices) == 0:
        return 0.0, h0
    
    x0 = h0[indices].copy()
    x0 = np.maximum(x0, 1e-10)
    
    def objective(x):
        h = np.zeros(N)
        h[indices] = x
        return -compute_ratio(h)
    
    bounds = [(0, 1)] * len(indices)
    
    try:
        result = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
        best_x = result.x
    except Exception:
        best_x = x0
    
    h_opt = np.zeros(N)
    h_opt[indices] = best_x
    
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    else:
        h_opt = np.ones(N) / N
        
    r = compute_ratio(h_opt)
    return r, h_opt

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_ratio = 0.0
    best_h = np.ones(N) / N
    
    # Initial best
    best_ratio = compute_ratio(best_h)
    
    # Try different sparsity levels
    # K=2 gives 1.25. K=3 gave 1.3125 in manual test.
    # Let's search K from 2 to 30.
    
    num_trials_per_k = 100
    
    for K in range(2, 31):
        if time.time() > deadline:
            break
            
        for trial in range(num_trials_per_k):
            if time.time() > deadline:
                break
            
            # Randomly select K indices
            # Bias towards clustered indices? Or random?
            # Random is good for exploration.
            indices = np.random.choice(N, size=K, replace=False)
            indices.sort()
            
            # Random initial values
            # Geometric or random?
            h0 = np.zeros(N)
            # Try a few different initialization strategies for the active indices
            if trial % 3 == 0:
                h0[indices] = np.random.rand(K)
            elif trial % 3 == 1:
                # Geometric
                r = np.random.uniform(0.5, 1.0)
                vals = r ** np.arange(K)
                h0[indices] = vals
            else:
                # Gaussian bump on indices
                x = np.linspace(0, 1, K)
                center = np.random.uniform(0.2, 0.8)
                width = np.random.uniform(0.1, 0.5)
                h0[indices] = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            
            # Optimize
            r, h_opt = optimize_sparse(h0, indices, max_iter=500)
            
            if r > best_ratio:
                best_ratio = r
                best_h = h_opt.copy()
                
    return list(best_h)
