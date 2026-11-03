import numpy as np
import time
import math

N = 100

def _log_energy(X):
    """Calculate logarithmic energy."""
    # Compute pairwise distances
    # X is (N, 3)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid log(0)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    # Energy is sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Sum over upper triangle
    iu = np.triu_indices(N, k=1)
    E = -np.sum(np.log(dist[iu]))
    return E

def _grad_energy(X):
    """Calculate gradient of logarithmic energy."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    # grad_{x_i} (-log d_ij) = (x_i - x_j) / d_ij^2
    # We want sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: dist_sq is d_ij^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # G_ij = (x_i - x_j) / d_ij^2
    # diff is (x_i - x_j)
    G = diff * inv_dist_sq[:, :, np.newaxis]
    grad = np.sum(G, axis=1)
    return grad

def _project_to_tangent(grad, X):
    """Project gradient onto tangent space."""
    # grad is (N, 3), X is (N, 3)
    # proj = grad - (grad . x) x
    dot = np.sum(grad * X, axis=1)
    proj = grad - dot[:, np.newaxis] * X
    return proj

def _step(X, grad, step_size):
    """Perform a gradient step and normalize."""
    X_new = X - step_size * grad
    # Normalize to unit sphere
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    # Handle zero norms just in case
    norms[norms == 0] = 1.0
    X_new = X_new / norms
    return X_new

def _optimize(X_init, time_limit):
    """Run gradient descent starting from X_init."""
    start_time = time.time()
    X = X_init.copy()
    E = _log_energy(X)
    step_size = 0.01
    min_step = 1e-6
    
    while time.time() - start_time < time_limit:
        grad = _grad_energy(X)
        proj_grad = _project_to_tangent(grad, X)
        
        X_new = _step(X, proj_grad, step_size)
        E_new = _log_energy(X_new)
        
        if E_new < E:
            X = X_new
            E = E_new
            # Can increase step size slightly if we are doing well?
            # step_size = min(step_size * 1.05, 0.1)
        else:
            step_size /= 2.0
            if step_size < min_step:
                break
                
    return X, E

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_X = None
    best_E = float('inf')
    
    # Define initial configurations
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    X_randoms = []
    for _ in range(9):
        # Random points on sphere
        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        X_randoms.append(v)
        
    all_starts = [X_fib] + X_randoms
    
    # Time budget per start
    # Total 115s. 10 starts. ~11s each.
    # But some might converge faster.
    # Let's allocate time dynamically or fixed. Fixed is safer for parallel-like logic in sequential code.
    time_per_start = deadline / len(all_starts)
    
    for X_init in all_starts:
        if time.time() - start_time > deadline:
            break
            
        # Refine
        X_opt, E_opt = _optimize(X_init, time_per_start)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    if best_X is None:
        return X_fib
        
    return best_X
