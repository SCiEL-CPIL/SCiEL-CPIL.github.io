"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient for points on the unit sphere.
    X: (N, 3) array of points.
    Returns: energy (float), grad (N, 3) array.
    """
    N = X.shape[0]
    # Compute pairwise distances
    # D[i, j] = ||X[i] - X[j]||
    # Using broadcasting
    # diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # D = np.linalg.norm(diff, axis=2)
    
    # More memory efficient for N=100 is fine with broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero for i==j
    np.fill_diagonal(D, 1.0) # Temporary, we will mask later
    
    # Energy: sum_{i<j} log(1/D[i,j]) = - sum_{i<j} log(D[i,j])
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Energy calculation
    # Use log(D)
    log_D = np.log(D)
    energy = -np.sum(log_D[mask])
    
    # Gradient calculation
    # grad_i = sum_{j != i} (X[j] - X[i]) / D[i,j]^2
    # Note: The gradient of -log(D) w.r.t X[i] is -(1/D) * (X[i]-X[j])/D = (X[j]-X[i])/D^2
    # We sum this over all j != i.
    
    # inv_D2 = 1 / D**2
    inv_D2 = 1.0 / (D**2)
    np.fill_diagonal(inv_D2, 0.0) # No self-interaction
    
    # grad[i] = sum_j diff[i,j] * inv_D2[i,j]
    # diff[i,j] = X[i] - X[j]
    # We want sum_j (X[j] - X[i]) / D[i,j]^2 = - sum_j (X[i] - X[j]) / D[i,j]^2
    # So grad = - np.sum(diff * inv_D2[:, :, np.newaxis], axis=1)
    
    grad = -np.sum(diff * inv_D2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def construct():
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin for evaluation overhead
    
    current_energy, current_grad = _energy_and_grad(X)
    best_energy = current_energy
    best_X = X.copy()
    
    # Gradient Descent with adaptive step size
    step_size = 0.1
    min_step_size = 1e-6
    max_step_size = 1.0
    
    # Pre-allocate for speed?
    # Loop for optimization
    for it in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        # Compute gradient
        current_energy, current_grad = _energy_and_grad(X)
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_X = X.copy()
            
        # Check convergence
        grad_norm = np.linalg.norm(current_grad)
        if grad_norm < 1e-8:
            break
            
        # Adaptive step size
        # Try a step
        X_new = X - step_size * current_grad
        X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
        
        new_energy, _ = _energy_and_grad(X_new)
        
        if new_energy < current_energy:
            X = X_new
            # Increase step size slightly if it worked well?
            # Or keep it steady. Let's keep it steady to be safe, or increase slowly.
            step_size = min(step_size * 1.05, max_step_size)
        else:
            # Decrease step size
            step_size *= 0.5
            if step_size < min_step_size:
                # Try a very small step or break
                break
                
    return best_X

# EVOLVE-BLOCK-END
