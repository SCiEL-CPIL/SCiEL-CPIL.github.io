import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    DELTA = 0.5 / N
    
    # Objective: Minimize max_conv / l2_sq
    # h is on the simplex (non-negative, sum to 1).
    # Parameterize with x, h = softmax(x)
    
    def objective(x):
        # Softmax
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        h = exp_x / np.sum(exp_x)
        
        # Calculate L2 squared
        l2_sq = np.sum(h**2)
        
        # Calculate Convolution Max
        # h is length N. h*h is length 2N-1.
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        
        # Avoid division by zero (though h is positive)
        if l2_sq < 1e-15:
            return 1e15
        if max_conv < 1e-15:
            return 1e15
            
        return max_conv / l2_sq

    # Initial guess: uniform
    x0 = np.zeros(N)
    
    # Try multiple random starts to avoid local minima
    best_val = np.inf
    best_h = None
    
    np.random.seed(42)
    for _ in range(100):
        x0 = np.random.randn(N) * 0.5
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000})
        
        x_max = np.max(res.x)
        exp_x = np.exp(res.x - x_max)
        h = exp_x / np.sum(exp_x)
        
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        val = max_conv / l2_sq
        
        if val < best_val:
            best_val = val
            best_h = h

    # Calculate the score for the best h found
    # Score = l2_sq / (l1 * DELTA * max_conv)
    # Since sum(h) = 1, l1 = 1 (in the discrete sum sense used in seed? No, seed uses l1=np.sum(h)).
    # Seed: l1 = np.sum(h). If sum is 1, l1=1.
    # Score = l2_sq / (1 * DELTA * max_conv) = (l2_sq / max_conv) / DELTA
    # We minimized max_conv / l2_sq. Let R = l2_sq / max_conv.
    # Score = R / DELTA.
    
    l2_sq = np.sum(best_h**2)
    h_conv = np.convolve(best_h, best_h, mode='full')
    max_conv = np.max(h_conv)
    
    score = (l2_sq / (np.sum(best_h) * DELTA * max_conv))
    
    return list(best_h)
