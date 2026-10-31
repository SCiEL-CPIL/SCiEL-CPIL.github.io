import numpy as np

N = 600

def construct():
    # Time limit guard
    import time
    start_time = time.time()
    deadline = 120  # seconds
    
    # Initialize with a cosine bump
    # Support is [-0.5, 0.5]
    # x values: -0.5, -0.5 + h, ...
    h = 1.0 / N
    # Centers of intervals
    # Interval i is [ -0.5 + i*h, -0.5 + (i+1)*h ]
    # Center: -0.5 + (i+0.5)*h
    centers = -0.5 + (np.arange(N) + 0.5) * h
    
    # Cosine bump: cos(pi * x) on [-0.5, 0.5]
    # cos(pi * -0.5) = 0, cos(pi * 0.5) = 0, cos(0) = 1
    x = np.cos(np.pi * centers)
    x = np.maximum(x, 0)  # Ensure non-negative
    
    # Normalize sum to 1
    if np.sum(x) > 0:
        x = x / np.sum(x)
    else:
        x = np.ones(N) / N
        
    best_x = x.copy()
    best_score = np.max(np.convolve(x, x))
    
    # Optimization loop
    # Minimize max of autoconvolution
    # Using soft-max gradient descent
    
    alpha = 5.0
    eta = 0.05
    T = 200
    
    for t in range(T):
        if time.time() > deadline - 10:
            break
            
        y = np.convolve(x, x, mode='full')
        
        # Compute soft-max weights
        # To avoid overflow, subtract max
        y_max = np.max(y)
        w = np.exp(alpha * (y - y_max))
        
        # Gradient of log-sum-exp of y wrt x
        # g_j = sum_k w_k * x_{k-j}
        # This is convolution of w and x
        # Note: y is length 2N-1. w is length 2N-1.
        # x is length N.
        # The gradient g will be length 2N-1, but we only need the central N parts?
        # Actually, x is defined on indices 0..N-1.
        # y_k = sum_{i=0}^{N-1} x_i x_{k-i} (with x_j=0 outside 0..N-1)
        # d y_k / d x_j = x_{k-j}
        # g_j = sum_k w_k x_{k-j}
        # This is the correlation of w and x?
        # Let's use np.convolve(w, x, mode='same')? 
        # w is centered at N-1. x is centered at N-1.
        # The convolution w * x will have the gradient at the center.
        
        g = np.convolve(w, x, mode='same')
        
        # Update x
        x = x - eta * g
        
        # Project to non-negative
        x = np.maximum(x, 0)
        
        # Normalize
        s = np.sum(x)
        if s < 1e-9:
            x = best_x.copy()
            break
        x = x / s
        
        # Check score
        y_new = np.convolve(x, x, mode='full')
        score_new = np.max(y_new)
        
        if score_new < best_score:
            best_score = score_new
            best_x = x.copy()
        else:
            # If it got worse, maybe reduce eta or revert
            # For now, just keep going, soft-max should guide it
            pass
            
    return best_x.tolist()
