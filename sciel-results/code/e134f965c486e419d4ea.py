import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Riemannian Gradient Descent
    # Energy: E = - sum_{i<j} log(|xi - xj|)
    # Gradient for point k: grad_k = - sum_{j != k} (xk - xj) / |xk - xj|^2
    
    # We will run for a fixed number of iterations or until time runs out.
    # Let's try 500 iterations.
    max_iters = 1000
    lr = 0.01
    decay = 0.999
    
    for t in range(max_iters):
        if time.time() > deadline:
            break
            
        # Compute pairwise distances
        # x is (N, 3)
        # diff is (N, N, 3)
        # dist_sq is (N, N)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Compute gradient
        # grad_k = - sum_{j != k} (xk - xj) / |xk - xj|^2
        # Note: diff[i, j] = xi - xj
        # term[i, j] = diff[i, j] / dist_sq[i, j]
        # grad_i = - sum_{j != i} term[i, j]
        
        # We want to sum over j for each i.
        # term shape (N, N, 3)
        # Divide diff by dist_sq[:, :, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = diff / dist_sq[:, :, np.newaxis]
        
        # Zero out the diagonal (i=j)
        idx = np.diag_indices(N)
        terms[idx] = 0.0
        
        # Sum over j (axis 1) to get gradient for each i
        grad = -np.sum(terms, axis=1)
        
        # Project gradient onto tangent space
        # grad_proj = grad - (grad . x) x
        dot = np.sum(grad * x, axis=1)[:, np.newaxis]
        grad_proj = grad - dot * x
        
        # Update
        x = x - lr * grad_proj
        
        # Normalize
        norms = np.linalg.norm(x, axis=1)[:, np.newaxis]
        x = x / norms
        
        # Decay learning rate
        lr *= decay
        
    return x
