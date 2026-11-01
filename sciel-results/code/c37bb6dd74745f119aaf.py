"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(points):
    """
    Calculate the logarithmic energy and its gradient for points on the sphere.
    points: (N, 3) array
    """
    n = points.shape[0]
    # Compute pairwise distances
    # points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero (points should be distinct, but safeguard)
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    # We sum over all i != j and divide by 2
    # log(1/d) = -log(d)
    # mask diagonal
    mask = ~np.eye(n, dtype=bool)
    log_dist = np.log(dist)
    # Sum over all pairs i!=j
    total_log_dist = np.sum(log_dist[mask])
    energy = -0.5 * total_log_dist
    
    # Gradient
    # grad_i = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    # diff is (N, N, 3), diff[i,j] = x_i - x_j
    # We want (x_j - x_i) / dist_sq[i,j]
    # So -diff / dist_sq[..., np.newaxis]
    
    # Zero out diagonal
    dist_sq_inv = np.zeros_like(dist_sq)
    dist_sq_inv[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = sum_j (x_j - x_i) / dist_sq[i,j]
    # Note: diff[i,j] = x_i - x_j. So x_j - x_i = -diff[i,j].
    # grad[i] = sum_j -diff[i,j] * dist_sq_inv[i,j]
    
    # Multiply each vector by its inverse distance squared
    # diff is (N, N, 3). dist_sq_inv is (N, N).
    # We need to multiply diff[i,j] by dist_sq_inv[i,j] for each k.
    weighted_diff = diff * dist_sq_inv[..., np.newaxis]
    
    # Sum over j (axis 1)
    # grad[i] = - sum_j weighted_diff[i,j]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _objective_and_grad(flat_points, mu):
    """
    Objective function with penalty for sphere constraint.
    flat_points: (3*N,) array
    mu: penalty strength
    """
    points = flat_points.reshape(N, 3)
    energy, grad = _energy_and_grad(points)
    
    # Penalty term: mu * sum(||x_i||^2 - 1)^2
    norms_sq = np.sum(points ** 2, axis=1)
    penalty = mu * np.sum((norms_sq - 1.0) ** 2)
    
    # Gradient of penalty
    # d/dx_i [ mu * (||x_i||^2 - 1)^2 ] = mu * 2 * (||x_i||^2 - 1) * 2 * x_i = 4 * mu * (||x_i||^2 - 1) * x_i
    penalty_grad = 4.0 * mu * (norms_sq - 1.0)[:, np.newaxis] * points
    
    total_obj = energy + penalty
    total_grad = grad + penalty_grad
    
    return total_obj, total_grad.flatten()

def _project_to_sphere(points):
    """
    Project points onto the unit sphere.
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-12)
    return points / norms

def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(seed):
    rng = np.random.RandomState(seed)
    points = rng.randn(N, 3)
    return _project_to_sphere(points)

def _icosahedral_init():
    # Generate points based on icosahedron vertices and rotations
    # This is a heuristic to provide a symmetric starting point
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    base = _project_to_sphere(base)
    # Duplicate and rotate to get N points
    # Simple approach: use Fibonacci as a fallback if this doesn't scale well
    # For N=100, we can just use Fibonacci or Random. 
    # Let's just return a random init with a specific seed for variety
    return _random_init(42)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    best_energy = float('inf')
    best_points = None
    
    # List of initializations
    inits = []
    inits.append(_fibonacci_init())
    for seed in range(10):
        inits.append(_random_init(seed))
        
    # Optimization parameters
    # L-BFGS-B is robust
    # We will try a few restarts
    
    for i, init_points in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Flatten points
        flat_init = init_points.flatten()
        
        # Try different penalty strengths
        # Start with moderate penalty
        mu_values = [10.0, 100.0]
        
        for mu in mu_values:
            if time.time() > deadline:
                break
                
            try:
                # Run L-BFGS-B
                res = minimize(
                    _objective_and_grad,
                    flat_init,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.status in [1, 2, 3]: # Convergence or max iter
                    opt_points = res.x.reshape(N, 3)
                    # Project to sphere to remove any residual drift
                    opt_points = _project_to_sphere(opt_points)
                    
                    # Calculate exact energy for comparison
                    energy, _ = _energy_and_grad(opt_points)
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_points = opt_points
                        
            except Exception:
                pass
    
    # If no optimization succeeded, return the best initial guess
    if best_points is None:
        best_points = _fibonacci_init()
    else:
        # Final polish: A few steps of projected gradient descent to ensure high precision
        # This helps with the "exact local solve" aspect
        points = best_points.copy()
        lr = 0.1
        for step in range(50):
            if time.time() > deadline:
                break
            energy, grad = _energy_and_grad(points)
            # Project gradient to tangent space
            # proj_grad = grad - (grad . x_i) x_i
            dots = np.sum(points * grad, axis=1, keepdims=True)
            tangent_grad = grad - dots * points
            # Update
            points = points - lr * tangent_grad
            # Project to sphere
            points = _project_to_sphere(points)
            # Simple backtracking line search to ensure decrease
            # (Omitted for brevity, simple fixed step with projection usually works well enough for final polish)
        
        # Check if polish improved energy
        final_energy, _ = _energy_and_grad(points)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = points

    return best_points

# EVOLVE-BLOCK-END
