"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize, differential_evolution

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = pts.shape[0]
    # Vectorized distance calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid log(0) and diagonal
    np.fill_diagonal(dist_sq, 1.0) 
    dist = np.sqrt(dist_sq)
    # We need sum over i < j. 
    # Sum of log(1/d) over all i!=j is 2 * sum over i<j.
    # So we can compute sum over all off-diagonals and divide by 2.
    # Or just use triu_indices.
    iu = np.triu_indices(n, 1)
    d = dist[iu]
    # Clamp to avoid log(0) if points are too close
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _gradient(pts):
    """Compute gradient of logarithmic energy w.r.t. point positions.
    
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    dE/dxi = -sum_{j!=i} (xi - xj) / |xi - xj|^2
    
    Vectorized implementation:
    grad[i] = - sum_{j!=i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    """
    n = pts.shape[0]
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal and very small distances
    # We want 1/dist_sq for j!=i, and 0 for i==j
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-20
    # Also mask diagonal explicitly just in case, though dist_sq[i,i]=0
    np.fill_diagonal(mask, False)
    
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = - sum_j inv_dist_sq[i, j] * diff[i, j]
    # diff[i, j] is vector (xi - xj).
    # We need to sum over axis 1 (j) for each i.
    # Multiply each diff[i,j,k] by inv_dist_sq[i,j] (broadcast over k)
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    # Sum over j (axis 1)
    grad = -np.sum(weighted_diff, axis=1)
    
    return grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    # Gradient descent with adaptive step size on the sphere
    # Using projected gradient: move in direction of negative gradient, then project to sphere
    lr = 0.01
    min_lr = 1e-6
    max_iter = 5000
    stagnation = 0
    window_size = 50
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute gradient
        g = _gradient(pts)
        
        # Take step
        pts_new = pts - lr * g
        
        # Project to sphere
        pts_new = _project_to_sphere(pts_new)
        
        # Check energy
        e_new = _energy(pts_new)
        
        if e_new < best_energy - 1e-12:
            best_pts = pts_new.copy()
            best_energy = e_new
            stagnation = 0
            # Increase learning rate slightly if improving
            lr = min(lr * 1.05, 0.05)
        else:
            pts = pts_new  # Keep the new position even if not improving, to explore
            stagnation += 1
            # Decrease learning rate
            lr *= 0.95
            if lr < min_lr:
                lr = min_lr
        
        # If stagnating too long, try a smaller perturbation
        if stagnation > window_size:
            # Add small random perturbation to escape local minima
            noise = 0.001 * np.random.randn(N, 3)
            pts_new = pts + noise
            pts_new = _project_to_sphere(pts_new)
            e_pert = _energy(pts_new)
            if e_pert < best_energy:
                pts = pts_new
                best_pts = pts_new.copy()
                best_energy = e_pert
                stagnation = 0
            else:
                pts = pts_new  # Keep perturbed position
            lr = min_lr * 2
            stagnation = 0
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        lr = 1e-5
        for iteration in range(2000):
            if time.time() > deadline:
                break
            g = _gradient(best_pts)
            best_pts = best_pts - lr * g
            best_pts = _project_to_sphere(best_pts)
            lr *= 0.99
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Vectorized check for very close points
    # We need to ensure no pair is closer than 1e-9
    # Compute all pairwise distances
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Find pairs too close
    iu = np.triu_indices(N, 1)
    close_mask = dist[iu] < 1e-9
    
    if np.any(close_mask):
        # Get indices of close pairs
        i_idx, j_idx = iu
        close_i = i_idx[close_mask]
        close_j = j_idx[close_mask]
        
        # Separate them
        for i, j in zip(close_i, close_j):
            best_pts[j] = best_pts[i] + 1e-6 * np.random.randn(3)
            best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
        
        # Re-project everything just in case
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
