import numpy as np

N = 100

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Optimization: Gradient Descent on the Sphere
    # Minimize E = - sum log(||x_i - x_j||)
    # Gradient of E is sum (x_j - x_i) / ||x_i - x_j||^2
    # Descent direction is sum (x_i - x_j) / ||x_i - x_j||^2
    
    max_iters = 3000
    eta = 0.05
    decay = 0.995
    momentum = 0.9
    
    V = np.zeros_like(X)
    
    for k in range(max_iters):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Handle i=j case
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Direction of steepest descent for energy minimization
        # This is -grad(E)
        G = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project to tangent space
        dot_G_X = np.sum(G * X, axis=1)
        G_tan = G - dot_G_X[:, np.newaxis] * X
        
        # Momentum update
        V = momentum * V + eta * G_tan
        X = X + V
        
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms
        
        # Decay
        eta *= decay
        
        # Convergence check
        if k > 50 and np.linalg.norm(V) < 1e-9:
            break
            
    # Final check for validity
    if not np.all(np.isfinite(X)):
        # Fallback to Fibonacci if something went wrong
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return X
