import numpy as np
import time
import math

N = 100

def _energy_and_grad(x):
    """
    Calculate energy and gradient for points on the sphere.
    x: (N, 3) array of points.
    Returns: energy, gradient (N, 3)
    """
    # Compute pairwise distances
    # d[i, j] = ||x[i] - x[j]||
    # To avoid zero division, we can add a small epsilon or handle it.
    # Since points are on the sphere, min distance is > 0 unless they coincide.
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist_sq, 1.0) # Temporary, will mask out
    
    dist = np.sqrt(dist_sq)
    
    # Gradient term: sum_{j != k} (x_j - x_k) / ||x_k - x_j||^2
    # Note: The force on k from j is (x_j - x_k) / dist^2.
    # We want to minimize E = - sum log(d). Grad E = - sum (x_k - x_j)/d^2 = sum (x_j - x_k)/d^2.
    # So we want to move in direction -Grad E = - sum (x_j - x_k)/d^2 = sum (x_k - x_j)/d^2.
    # Wait.
    # E = - sum_{i<j} log(d_ij)
    # dE/dx_k = - sum_{j!=k} d/dx_k log(d_kj) = - sum_{j!=k} (1/d_kj) * (x_k - x_j)/d_kj = - sum_{j!=k} (x_k - x_j)/d_kj^2.
    # So Gradient of Energy is G_k = - sum_{j!=k} (x_k - x_j)/d_kj^2.
    # We want to minimize E, so we move in direction -G_k.
    # -G_k = sum_{j!=k} (x_k - x_j)/d_kj^2.
    
    # Let's compute the vector sum for each k.
    # term[k, j] = (x_k - x_j) / dist_sq[k, j]
    # This is a (N, N, 3) tensor.
    
    # We can compute this more efficiently.
    # diff is (N, N, 3). diff[k, j] = x_k - x_j.
    # inv_dist_sq is (N, N).
    # forces = diff * inv_dist_sq[:, :, np.newaxis]
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    forces = diff * inv_dist_sq[:, :, np.newaxis]
    
    # Gradient of Energy G_k = - sum_j forces[k, j]
    # Actually, forces[k, j] is (x_k - x_j)/d^2.
    # Sum over j gives sum (x_k - x_j)/d^2.
    # This is exactly -G_k (the descent direction).
    # So Descent Direction D_k = sum_j (x_k - x_j)/d^2.
    
    grad_energy = -np.sum(forces, axis=1)
    
    # Energy calculation
    # E = - sum_{i<j} log(d_ij)
    # Use log(dist)
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    log_dists = np.log(dist[mask])
    energy = -np.sum(log_dists) / 2.0 # Because sum over all i!=j is double sum over i<j
    
    return energy, grad_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_energy, grad = _energy_and_grad(x)
    best_energy = current_energy
    best_x = x.copy()
    
    # Gradient Descent on Sphere
    # Step size decay
    alpha = 0.01
    min_alpha = 1e-6
    decay = 0.999
    
    for step in range(1000000): # Max steps
        if time.time() > deadline:
            break
        
        # Project gradient to tangent plane
        # T_k = G_k - (G_k . x_k) x_k
        dot = np.sum(x * grad, axis=1)
        tangent_grad = grad - dot[:, np.newaxis] * x
        
        # Update
        x = x + alpha * tangent_grad
        
        # Normalize
        norms = np.linalg.norm(x, axis=1)
        x = x / norms[:, np.newaxis]
        
        # Calculate new energy and gradient
        new_energy, grad = _energy_and_grad(x)
        
        # Check for improvement
        if new_energy < current_energy:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = x.copy()
        else:
            # If energy increases, reduce step size
            alpha *= 0.5
            if alpha < min_alpha:
                # Try a small random perturbation to escape local minima?
                # Or just stop if alpha is too small
                pass
        
        # Decay step size
        alpha *= decay
        if alpha < min_alpha:
            alpha = min_alpha

    return best_x
