import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    # Allow up to 115 seconds, leaving 5s for safety
    deadline = start_time + 115.0
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Momentum
    V = np.zeros_like(X)
    lr = 0.005
    momentum = 0.9
    
    for step in range(100000):
        if time.time() > deadline:
            break
            
        # Compute squared distances
        D = (X**2).sum(1)[:, None] + (X**2).sum(1)[None, :] - 2 * (X @ X.T)
        # Avoid negative zeros or tiny negatives due to floating point
        D = np.maximum(D, 1e-12)
        np.fill_diagonal(D, 1.0)
        
        # W_ij = 1 / D_ij for i != j
        W = 1.0 / D
        np.fill_diagonal(W, 0.0)
        
        # Gradient calculation
        S = W.sum(1) # (N,)
        WX = W @ X # (N, 3)
        G = -S[:, None] * X + WX
        
        # Update
        V = momentum * V - lr * G
        X = X + V
        
        # Project to sphere
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms
        
        # Adaptive learning rate
        min_dist = np.sqrt(np.min(D))
        if min_dist < 0.05:
            lr *= 0.8
        elif min_dist > 0.2 and lr < 0.005:
            lr *= 1.05
            
    return X
