import numpy as np
import time

N = 150

def log_energy(points):
    """Compute the logarithmic energy."""
    # points: (N, 3)
    # distances: (N, N)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Avoid log(0)
    dists = np.maximum(dists, 1e-12)
    # Energy is sum_{i<j} log(1/d_{ij}) = - sum_{i<j} log(d_{ij})
    # We only need upper triangle
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dists[iu]))

def log_energy_grad(points):
    """Compute the gradient of the logarithmic energy."""
    # points: (N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists_sq = np.maximum(dists_sq, 1e-24) # Avoid division by zero
    dists = np.sqrt(dists_sq)
    
    # Gradient of -log(d_{ij}) wrt x_i is - (x_i - x_j) / d_{ij}^2
    # grad_i = - sum_{j != i} (x_i - x_j) / d_{ij}^2
    # Note: diff[i,j] = x_i - x_j
    # So we need sum_{j != i} diff[i,j] / dists_sq[i,j]
    
    # Create a matrix of 1/dists_sq, with 0 on the diagonal
    inv_dists_sq = np.zeros_like(dists_sq)
    np.fill_diagonal(inv_dists_sq, 0)
    np.divide(1.0, dists_sq, out=inv_dists_sq, where=(dists_sq > 1e-24))
    
    # grad = - diff * inv_dists_sq, summed over j
    # diff is (N, N, 3)
    # inv_dists_sq is (N, N)
    # We need to multiply each diff[i,j,k] by inv_dists_sq[i,j]
    grad = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    return grad

def project_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return points / norms

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_energy = log_energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Gradient descent with projection
    step_size = 0.01
    min_step_size = 1e-6
    
    while time.time() < deadline:
        grad = log_energy_grad(points)
        
        # Take a step
        new_points = points - step_size * grad
        new_points = project_sphere(new_points)
        
        new_energy = log_energy(new_points)
        
        if new_energy < current_energy:
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
            # Increase step size slightly
            step_size = min(step_size * 1.05, 0.1)
        else:
            # Decrease step size
            step_size *= 0.5
            if step_size < min_step_size:
                break
        
        # Check time
        if time.time() > deadline:
            break
            
    # Final polish with smaller steps if time allows
    if time.time() < deadline - 10:
        step_size = 1e-4
        while time.time() < deadline - 5:
            grad = log_energy_grad(best_points)
            new_points = best_points - step_size * grad
            new_points = project_sphere(new_points)
            new_energy = log_energy(new_points)
            if new_energy < best_energy:
                best_points = new_points
                best_energy = new_energy
            else:
                step_size *= 0.5
                if step_size < 1e-7:
                    break
                    
    return best_points
