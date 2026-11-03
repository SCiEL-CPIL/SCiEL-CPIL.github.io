import numpy as np
import time

N = 100

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    deadline = 110.0  # Leave 10s buffer for 120s limit
    
    # Parameters
    lr = 1e-4
    momentum = 0.9
    v = np.zeros_like(X)
    
    best_X = X.copy()
    best_E = np.inf
    
    def energy_and_grad(X):
        # Compute pairwise distances
        # D[i, j] = ||x_i - x_j||
        # Use broadcasting
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero for i==j
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Energy: sum_{i<j} log(1/dist_ij)
        # Only upper triangle
        iu = np.triu_indices(N, 1)
        E = np.sum(-np.log(dist[iu]))
        
        # Gradient
        # grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
        # Note: dist_ij^2 = dist_sq[i,j]
        # Force vector for i: sum_{j != i} (x_i - x_j) / dist_sq[i,j]
        # This is -grad E.
        # We want to minimize E, so we move in direction -grad E.
        # So update: X += lr * force
        
        # force[i] = sum_{j != i} (X[i] - X[j]) / dist_sq[i,j]
        # diff[i,j] = X[i] - X[j]
        # factor[i,j] = 1 / dist_sq[i,j]
        
        # We can compute this efficiently
        # inv_dist_sq = 1.0 / dist_sq
        # np.fill_diagonal(inv_dist_sq, 0.0)
        # force = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        force = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        return E, force

    E, force = energy_and_grad(X)
    best_E = E
    
    # Gradient Descent with Momentum
    while time.time() < deadline:
        # Update velocity
        v = momentum * v + force
        # Update position
        X_new = X + lr * v
        
        # Project to sphere
        norms = np.linalg.norm(X_new, axis=1)
        X_new = X_new / norms[:, np.newaxis]
        
        # Check energy
        E_new, force_new = energy_and_grad(X_new)
        
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
            
        # Adaptive learning rate (simple)
        # If energy increased, reduce lr
        if E_new > E:
            lr *= 0.8
            # Revert to previous X if step was bad? 
            # Or just keep X and reduce lr.
            # Let's keep X_new if it's not too bad, but reduce lr.
            # Actually, if E_new > E, it's a bad step.
            # Let's revert X to X and reduce lr.
            X = X # revert
            v = 0.0 # reset momentum?
            # But force is for X_new. We need force for X.
            # We already have force for X from previous iter.
            # So we can just continue with old force and new lr.
            # But X is unchanged.
        else:
            X = X_new
            force = force_new
            E = E_new
            # Optionally increase lr slightly if improving steadily?
            # lr *= 1.01
    
    return best_X
