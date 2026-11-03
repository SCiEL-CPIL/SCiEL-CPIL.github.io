import numpy as np
import time
import math

N = 150

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_gradient(points):
    # points: (N, 3) array
    # Ensure points are on the sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    x = points / norms
    
    # Compute pairwise distances
    # d_ij = ||x_i - x_j||
    # Using broadcasting
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid log(0)
    dist_sq = np.maximum(dist_sq, 1e-18)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # We sum over all i != j and divide by 2, or just use upper triangle
    # Using upper triangle for efficiency and correctness
    iu = np.triu_indices(N, k=1)
    d_pairs = dist[iu]
    
    # Handle very small distances
    d_pairs = np.maximum(d_pairs, 1e-9)
    
    E = -np.sum(np.log(d_pairs))
    
    # Gradient
    # grad_i E = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: d_ij^2 = dist_sq[i,j]
    # Term: (x_i - x_j) / dist_sq[i,j]
    
    # We need to be careful with the diagonal
    # diff[i,j] = x_i - x_j
    # grad_i = - sum_j (x_i - x_j) / dist_sq[i,j] for j != i
    
    # Create a mask for j != i
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # dist_sq with diagonal set to 1 to avoid div by zero in calculation, though masked
    dist_sq_safe = dist_sq.copy()
    np.fill_diagonal(dist_sq_safe, 1.0)
    
    # Weighted sum of differences
    # W[i,j] = 1 / dist_sq[i,j] if i!=j else 0
    W = np.where(mask, 1.0 / dist_sq_safe, 0.0)
    
    # grad_i = - sum_j W[i,j] * (x_i - x_j)
    # = - sum_j W[i,j] * x_i + sum_j W[i,j] * x_j
    # = - x_i * sum_j W[i,j] + sum_j W[i,j] * x_j
    
    row_sums = np.sum(W, axis=1)
    grad_euclid = -x * row_sums[:, np.newaxis] + W @ x
    
    # Project gradient to tangent space
    # g_tan = g_euclid - (g_euclid . x) x
    dot = np.sum(grad_euclid * x, axis=1, keepdims=True)
    grad_tan = grad_euclid - dot * x
    
    return E, grad_tan.flatten()

def _objective(x):
    return _energy_and_gradient(x)[0]

def _jac(x):
    return _energy_and_gradient(x)[1]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial points
    x0 = _fibonacci_sphere(N)
    
    # Reshape to 1D for optimizer
    x0_flat = x0.flatten()
    
    best_x = x0_flat.copy()
    best_energy = _objective(best_x)
    
    try:
        from scipy.optimize import minimize
        
        # L-BFGS-B is efficient for this size
        # We use the projected gradient trick:
        # fun and jac normalize the input points.
        # The optimizer moves in R^{3N}, but we interpret the result on the sphere.
        
        res = minimize(
            _objective,
            x0_flat,
            jac=_jac,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_energy:
            best_x = res.x
            best_energy = res.fun
            
        # Check time
        if time.time() > deadline:
            pass # Just return what we have
            
        # Optional: A second pass with tighter tolerance if time permits
        if time.time() < deadline - 20.0:
             res2 = minimize(
                _objective,
                best_x,
                jac=_jac,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-10}
            )
             if res2.fun < best_energy:
                 best_x = res2.x
                 best_energy = res2.fun
                 
    except Exception:
        pass
        
    # Reshape back to (N, 3) and normalize one last time
    x_final = best_x.reshape(N, 3)
    norms = np.linalg.norm(x_final, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x_final = x_final / norms
    
    return x_final
