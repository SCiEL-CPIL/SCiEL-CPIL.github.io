import numpy as np

N = 400

def construct():
    # Sparse combinatorial design approach
    # Select a small subset of intervals (support)
    K = 20
    support = np.arange(K)
    
    # Initialize with a signed block pattern
    x = np.zeros(N)
    x[support[:K//2]] = 1.0
    x[support[K//2:]] = -1.0
    
    # Normalize to L2 = 1
    x = x / np.sqrt(np.sum(x**2))
    
    # Simple local optimization: flip signs to minimize L3 norm of autoconvolution
    # We only optimize the signs within the support
    best_x = x.copy()
    best_score = np.sum(np.abs(np.convolve(best_x, best_x, mode='full'))**3)
    
    # Limit iterations for time safety
    for _ in range(100):
        improved = False
        for idx in support:
            # Flip sign
            x[idx] = -x[idx]
            score = np.sum(np.abs(np.convolve(x, x, mode='full'))**3)
            if score < best_score:
                best_score = score
                best_x = x.copy()
                improved = True
            else:
                x[idx] = -x[idx] # Revert
        
        if not improved:
            break
            
    return best_x
