import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad_fast(points):
    """
    Compute the logarithmic energy and its gradient.
    points: (N, 3) array of unit vectors.
    Returns: energy (float), grad (N, 3) array.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero (diagonal)
    dist_sq_safe = dist_sq.copy()
    np.fill_diagonal(dist_sq_safe, 1.0)
    
    dist = np.sqrt(dist_sq_safe)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Gradient for each i: sum_{j != i} (x_i - x_j) / d_ij^2
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq_safe[mask]
    
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy: sum_{i<j} -log(d_ij)
    log_d = np.zeros_like(dist)
    log_d[mask] = np.log(dist[mask])
    energy = -0.5 * np.sum(log_d[mask])
    
    return energy, grad

def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _tangent_project(points, grad):
    """Project gradient to tangent space of the sphere."""
    # For points on unit sphere, tangent space is orthogonal to the point
    # grad_tangent = grad - (grad . x) * x
    grad_dot_x = np.sum(grad * points, axis=1, keepdims=True)
    return grad - grad_dot_x * points

def _sphere_fun_and_grad(flat_points):
    """
    Objective and gradient for scipy minimize.
    flat_points: (3N,) array of flattened points.
    """
    points = flat_points.reshape(N, 3)
    points = _project_to_sphere(points)
    
    energy, grad = _energy_and_grad_fast(points)
    grad_tangent = _tangent_project(points, grad)
    
    return energy, grad_tangent.flatten()

def _optimize_lbfgs(points, max_iter=500, time_limit=30.0):
    """
    Optimize using L-BFGS-B with tangent space projection.
    """
    flat_points = points.flatten()
    
    result = minimize(
        _sphere_fun_and_grad,
        flat_points,
        jac=True,
        method='L-BFGS-B',
        options={
            'maxiter': max_iter,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxfun': max_iter * 2,
        }
    )
    
    optimized_points = result.x.reshape(N, 3)
    optimized_points = _project_to_sphere(optimized_points)
    
    return optimized_points, result.fun

def _polish(points, max_iter=1000, time_limit=20.0):
    """
    High-precision polish using gradient descent in tangent space.
    """
    start_time = time.time()
    lr = 0.001
    momentum = 0.9
    v = np.zeros_like(points)
    
    for step in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad_fast(points)
        grad_tangent = _tangent_project(points, grad)
        
        v = momentum * v - lr * grad_tangent
        points = points + v
        points = _project_to_sphere(points)
        
        # Reduce learning rate over time
        lr *= 0.999
    
    return points

def construct():
    start_time = time.time()
    timeout = 140.0  # Leave 10s margin for evaluation overhead (150s total)
    
    best_energy = float('inf')
    best_points = None
    
    # Initial configuration: Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # First: L-BFGS-B optimization from Fibonacci start
    if time.time() - start_time < timeout - 10:
        points, energy = _optimize_lbfgs(points, max_iter=300, time_limit=timeout - 10 - (time.time() - start_time))
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
    
    # Polish the result
    if time.time() - start_time < timeout - 5:
        points = _polish(points, max_iter=500, time_limit=min(20.0, timeout - 5 - (time.time() - start_time)))
        energy, _ = _energy_and_grad_fast(points)
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
    
    # Random restarts with L-BFGS-B
    rng = np.random.RandomState(42)
    num_restarts = 5
    restart_time_budget = (timeout - (time.time() - start_time)) / max(1, num_restarts)
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout - 5:
            break
        
        # Generate random points
        points = rng.randn(N, 3)
        points = _project_to_sphere(points)
        
        # L-BFGS-B optimization for this random start
        remaining_time = timeout - 5 - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        points, energy = _optimize_lbfgs(points, max_iter=200, time_limit=min(15.0, remaining_time - 5))
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Polish if this was an improvement
        if energy < best_energy and time.time() - start_time < timeout - 3:
            points = _polish(points, max_iter=300, time_limit=min(10.0, timeout - 3 - (time.time() - start_time)))
            energy, _ = _energy_and_grad_fast(points)
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
    
    # Final polish on the best result
    if best_points is not None and time.time() - start_time < timeout - 2:
        final_points = _polish(best_points, max_iter=500, time_limit=min(15.0, timeout - 2 - (time.time() - start_time)))
        final_energy, _ = _energy_and_grad_fast(final_points)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    # Final normalization
    best_points = _project_to_sphere(best_points)
    
    return best_points
