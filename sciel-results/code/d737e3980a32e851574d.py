import numpy as np
import time
import math

N = 100

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    X: (N, 3) array of points.
    Returns: energy (float), grad (N, 3) array.
    """
    # Compute pairwise squared distances
    # X[:, np.newaxis, :] - X[np.newaxis, :, :]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i == j
    # We only care about i != j
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    # sq_dist is d^2, so log(d) = 0.5 * log(sq_dist)
    # We sum over i < j
    triu_indices = np.triu_indices(N, k=1)
    d_sq = sq_dist[triu_indices]
    
    # Clip to avoid log(0) if points coincide (shouldn't happen with good init/step)
    d_sq = np.clip(d_sq, 1e-15, None)
    
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient:
    # dE/dx_i = sum_{j != i} - (x_i - x_j) / ||x_i - x_j||^2
    # Let's compute the full gradient matrix
    # G_i = sum_{j != i} - diff[i, j] / sq_dist[i, j]
    
    # diff is (N, N, 3)
    # sq_dist is (N, N)
    # We need to divide diff by sq_dist[..., np.newaxis]
    # But we must mask the diagonal to avoid division by zero
    
    inv_sq_dist = np.zeros_like(sq_dist)
    inv_sq_dist[mask] = 1.0 / sq_dist[mask]
    
    # G[i] = sum_j - diff[i,j] * inv_sq_dist[i,j]
    # diff[i,j] = x_i - x_j
    # So G[i] = - sum_j (x_i - x_j) / sq_dist[i,j]
    
    # We can use einsum or matrix multiplication
    # G = - np.sum(diff * inv_sq_dist[..., np.newaxis], axis=1)
    # Note: diff[i,j] is x_i - x_j.
    # The term for j in the sum for i is (x_i - x_j)/d^2.
    # The term for i in the sum for j is (x_j - x_i)/d^2 = -(x_i - x_j)/d^2.
    # This is consistent.
    
    grad = -np.sum(diff * inv_sq_dist[..., np.newaxis], axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X.copy()
    best_E, _ = _energy_and_grad(X)
    
    # Gradient Descent with Momentum and Adaptive Step Size
    # Parameters
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(X)
    
    # We will run for a fixed number of iterations or until time limit
    # N=100 is small, so we can do many iterations.
    max_iters = 20000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        E, G = _energy_and_grad(X)
        
        # Project gradient to tangent space
        # G_tangent = G - (G . X) X
        dot = np.sum(G * X, axis=1, keepdims=True)
        G_tangent = G - dot * X
        
        # Momentum update
        v = momentum * v - lr * G_tangent
        X_new = X + v
        
        # Normalize to sphere
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        
        # Check for improvement
        E_new, _ = _energy_and_grad(X_new)
        
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
            
        # If energy increased, we might need to reduce lr or accept if it's a small increase (not needed for simple GD)
        # For simplicity, we just update if it's better, or we can do backtracking.
        # Let's just update X if it's better, otherwise keep old X and reduce lr?
        # Actually, standard GD updates X regardless, but if it diverges, we lose.
        # With projection and small lr, it's usually stable.
        # Let's use a simple backtracking line search if energy increases.
        
        if E_new > E:
            # Backtracking
            alpha = 1.0
            X_try = X
            while alpha > 1e-6:
                X_try = X + alpha * v
                norms_try = np.linalg.norm(X_try, axis=1, keepdims=True)
                X_try = X_try / norms_try
                E_try, _ = _energy_and_grad(X_try)
                if E_try < E:
                    X = X_try
                    break
                alpha *= 0.5
            else:
                # If we didn't find a good step, reduce learning rate
                lr *= 0.5
                # Keep X as is
                continue
                
        X = X_new
        
    return best_X
