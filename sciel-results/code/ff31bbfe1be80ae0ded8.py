import numpy as np
import math
from scipy.optimize import minimize

N = 100

def compute_energy(points):
    """Compute the logarithmic energy of the points."""
    # points is (N, 3)
    # We want sum_{i<j} log(1/|xi - xj|) = - sum_{i<j} log(|xi - xj|)
    # Using numpy for speed during optimization
    # Calculate all pairwise distances
    # dists[i, j] = ||points[i] - points[j]||
    # We only need i < j
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dists = dists[mask]
    # Avoid log(0)
    dists = np.maximum(dists, 1e-12)
    return -np.sum(np.log(dists))

def compute_gradient(points):
    """Compute the gradient of the energy with respect to the points."""
    # grad[k] = sum_{j != k} (points[k] - points[j]) / ||points[k] - points[j]||^2
    grad = np.zeros_like(points)
    # Vectorized calculation
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dist_sq[i, j] = ||points[i] - points[j]||^2
    dist_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    # We want sum_j (points[i] - points[j]) / dist_sq[i, j]
    # This is equivalent to sum_j diff[i, j] / dist_sq[i, j]
    # Divide each row by dist_sq
    # We need to handle the diagonal carefully, but since diff[i,i]=0, it's 0 anyway.
    inv_dist_sq = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_dist_sq, 0.0)
    # Only compute for i != j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    # Actually, the gradient formula is symmetric in a way, but let's just sum over all j != i.
    # The term for j is (x_i - x_j) / ||x_i - x_j||^2.
    # The term for i in the sum for x_j is (x_j - x_i) / ||x_j - x_i||^2 = -(x_i - x_j) / ||x_i - x_j||^2.
    # So the total force on x_i is sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2.
    
    # Let's compute the sum directly.
    # We can use broadcasting.
    # diff is (N, N, 3)
    # dist_sq is (N, N)
    # We want to sum over axis 1.
    # term[i, j, :] = diff[i, j, :] / dist_sq[i, j]
    # We need to avoid division by zero on the diagonal.
    # Since diff[i, i, :] is 0, we can just set dist_sq[i, i] to 1.
    # But we already did that.
    # So term = diff / dist_sq[:, :, np.newaxis]
    term = diff / dist_sq[:, :, np.newaxis]
    # Sum over j (axis 1)
    grad = np.sum(term, axis=1)
    return grad

def penalty_energy(x, penalty_weight=1e4):
    points = x.reshape(-1, 3)
    energy = compute_energy(points)
    # Penalty for not being on the sphere
    norms_sq = np.sum(points**2, axis=1)
    penalty = penalty_weight * np.sum((norms_sq - 1.0)**2)
    return energy + penalty

def penalty_gradient(x, penalty_weight=1e4):
    points = x.reshape(-1, 3)
    grad_energy = compute_gradient(points)
    # Gradient of penalty: 2 * penalty_weight * (||x_i||^2 - 1) * x_i
    norms_sq = np.sum(points**2, axis=1)
    grad_penalty = 2.0 * penalty_weight * (norms_sq - 1.0)[:, np.newaxis] * points
    return (grad_energy + grad_penalty).flatten()

def optimize(points, max_iters=500):
    x0 = points.flatten()
    res = minimize(penalty_energy, x0, jac=penalty_gradient, method='L-BFGS-B', 
                   options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
    return res.x.reshape(-1, 3)

def construct():
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try a few different initializations and keep the best
    best_points = points_fib
    best_energy = compute_energy(points_fib)
    
    # Try random initialization
    np.random.seed(42)
    for restart in range(10):
        if restart == 0:
            start_points = points_fib
        else:
            # Random points on the sphere
            vecs = np.random.randn(N, 3)
            start_points = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        
        # Optimize
        opt_points = optimize(start_points, max_iters=500)
        # Normalize again to ensure they are on the sphere
        opt_points = opt_points / np.linalg.norm(opt_points, axis=1, keepdims=True)
        
        energy = compute_energy(opt_points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points
            
    return best_points
