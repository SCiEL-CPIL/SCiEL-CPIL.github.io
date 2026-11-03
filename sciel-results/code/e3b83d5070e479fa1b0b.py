import numpy as np
import time
import math

N = 100

def fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def compute_energy_and_grad(points):
    """
    Computes the logarithmic energy and its gradient for points on the unit sphere.
    Energy E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    Gradient g_i = - sum_{j!=i} (x_i - x_j) / d_ij^2
    """
    n = points.shape[0]
    # Pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # Squared distances
    d2 = np.sum(diff * diff, axis=2)
    # Avoid division by zero (diagonal)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    # Energy: sum_{i<j} -log(d_ij)
    # We can compute sum over all i!=j and divide by 2
    log_d = np.log(d)
    np.fill_diagonal(log_d, 0.0)
    energy = -0.5 * np.sum(log_d)
    
    # Gradient
    # g_i = - sum_{j!=i} (x_i - x_j) / d_ij^2
    # Note: diff[i,j] = x_i - x_j
    # term[i,j] = diff[i,j] / d2[i,j]
    # grad[i] = - sum_j term[i,j]
    
    # To avoid issues with diagonal, we set d2 diagonal to 1, but term should be 0 on diagonal.
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # term[i,j] = diff[i,j] * inv_d2[i,j]
    # grad[i] = - sum_j diff[i,j] * inv_d2[i,j]
    # Using matrix multiplication:
    # grad = - (diff * inv_d2[:, :, np.newaxis]).sum(axis=1)
    
    weights = diff * inv_d2[:, :, np.newaxis]
    grad = -np.sum(weights, axis=1)
    
    return energy, grad

def project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def riemannian_gradient(grad, points):
    """
    Projects the Euclidean gradient onto the tangent space of the sphere at each point.
    Tangent space at x is {v | v . x = 0}.
    Projection of g onto tangent space: g - (g . x) x
    """
    dot = np.sum(grad * points, axis=1, keepdims=True)
    return grad - dot * points

def optimize(points, max_iters=5000, lr_start=0.01, lr_end=1e-6):
    points = points.copy()
    best_energy = np.inf
    best_points = points.copy()
    
    # Precompute schedule
    # Linear decay or exponential? Exponential is often better for optimization.
    # Let's use a simple decay.
    
    for i in range(max_iters):
        # Learning rate decay
        # t is in [0, 1]
        t = i / max_iters
        # lr = lr_start * (lr_end / lr_start) ** t
        lr = lr_start * (lr_end / lr_start) ** t
        
        energy, grad = compute_energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            
        # Riemannian gradient
        r_grad = riemannian_gradient(grad, points)
        
        # Update
        points = points - lr * r_grad
        
        # Project back to sphere
        points = project_to_sphere(points)
        
    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initializations
    # 1. Fibonacci
    p0 = fibonacci_spiral(N)
    
    # 2. Perturbed Fibonacci
    rng = np.random.default_rng(42)
    p1 = p0 + rng.normal(0, 0.01, (N, 3))
    p1 = project_to_sphere(p1)
    
    # 3. Another perturbation
    p2 = p0 + rng.normal(0, 0.05, (N, 3))
    p2 = project_to_sphere(p2)
    
    candidates = [p0, p1, p2]
    
    best_points = p0
    best_energy = np.inf
    
    for p_init in candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Run optimization
        # We can run multiple rounds with different LR schedules
        # Round 1: Coarse
        p_opt, e_opt = optimize(p_init, max_iters=2000, lr_start=0.05, lr_end=1e-4)
        
        if time.time() > deadline - 5.0:
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = p_opt
            break
            
        # Round 2: Fine polish
        p_opt, e_opt = optimize(p_opt, max_iters=2000, lr_start=1e-4, lr_end=1e-7)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_points = p_opt
            
    return best_points
