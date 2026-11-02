import numpy as np
import time
import math

N = 100

def _compute_energy_and_grad(points):
    """
    Computes the logarithmic energy and its gradient.
    Energy E = sum_{i<j} log(1 / dist_ij) = - sum_{i<j} log(dist_ij)
    We want to minimize E, which is equivalent to maximizing sum_{i<j} log(dist_ij).
    
    Gradient of E w.r.t x_i:
    dE/dx_i = - sum_{j != i} d/dx_i log(||x_i - x_j||)
            = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
            
    So we perform gradient descent on E, which is gradient ascent on -E.
    Update: x_i <- x_i - eta * grad_E_i
           = x_i + eta * sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # points shape (N, 3)
    # dists shape (N, N)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    # We only care about i != j
    # Create a mask for i != j
    mask = ~np.eye(N, dtype=bool)
    
    # For energy: sum over i<j
    # dists is symmetric.
    # Sum of log(1/d) for i<j
    # We can sum all i!=j and divide by 2.
    # log(1/d) = -log(d)
    # Sum_{i!=j} -log(d_ij) / 2
    # Note: dists[i,i] is 0, log(0) is -inf. We mask them out.
    
    # For gradient:
    # grad_i = - sum_{j!=i} (x_i - x_j) / dist_ij^2
    # Note: (x_i - x_j) / dist_ij^2 is the vector from j to i divided by dist^2.
    # This is the repulsive force.
    
    # Let's compute the repulsive force term: F_ij = (x_i - x_j) / dist_ij^2
    # We need to handle dist=0.
    # If dist is very small, the force is huge.
    # We can replace 0 with a small epsilon in the denominator.
    
    dists_safe = dists.copy()
    dists_safe[~mask] = 1.0 # Avoid div by 0 for diagonal, doesn't matter as we mask later
    # Actually, we want to compute (x_i - x_j) / dist^2 for all i != j.
    
    # Vectorized calculation of forces
    # diff is (N, N, 3)
    # dists is (N, N)
    # inv_dist_sq = 1 / dists^2
    inv_dist_sq = np.zeros_like(dists)
    np.divide(1.0, dists_safe**2, out=inv_dist_sq, where=mask)
    
    # Force on i from j is diff[i,j] * inv_dist_sq[i,j]
    # diff[i,j] = x_i - x_j
    forces = diff * inv_dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    grad_E = -np.sum(forces, axis=1)
    
    # Energy
    # Sum of -log(d) for i<j
    # We can sum -log(dists) over all i!=j and divide by 2
    # Mask out diagonal
    log_dists = np.zeros_like(dists)
    np.log(dists_safe, out=log_dists, where=mask)
    # Note: where=mask means we compute log where mask is True.
    # But np.log with where parameter might leave uncomputed values as 0 or garbage?
    # Better to compute on masked array or use np.where.
    # Since dists_safe is 1.0 on diagonal, log(1)=0. So it's fine.
    energy = -0.5 * np.sum(log_dists)
    
    return energy, grad_E

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    current_energy, _ = _compute_energy_and_grad(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Gradient Ascent on -Energy (i.e., Gradient Descent on Energy)
    # Update: points = points - eta * grad_E
    # which is points + eta * repulsive_force
    
    # We can try a few different step sizes or restarts
    # But N=100 is small enough that one good run might suffice if step size is tuned.
    # Let's use a fixed step size with decay or just a fixed small step.
    # A common choice is eta = 0.01 or 0.001.
    
    # Let's try a simple iterative refinement
    # We will run for a maximum number of iterations or until time limit.
    
    max_iters = 5000
    eta = 0.05
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        energy, grad = _compute_energy_and_grad(points)
        
        # If energy is NaN or Inf, break
        if not np.isfinite(energy):
            break
            
        # Update
        points = points - eta * grad
        
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-12] = 1.0
        points = points / norms
        
        # Check improvement
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            
    return best_points
