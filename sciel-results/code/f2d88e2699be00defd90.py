"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere."""
    n = pts.shape[0]
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(n, 1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient: d/dx_i sum_{j!=i} log(1/|x_i - x_j|) = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    # = sum_{j!=i} (x_j - x_i)/|x_i - x_j|^2
    # For each pair (i,j), contribution to grad_i is (x_j - x_i)/d^2, to grad_j is (x_i - x_j)/d^2
    inv_d2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_d2, 0.0)
    # grad[i] = sum_j (x_j - x_i) / |x_i - x_j|^2
    # = sum_j x_j / d^2 - x_i * sum_j 1/d^2
    sum_xj_over_d2 = inv_d2 @ pts  # shape (n, 3)
    sum_inv_d2 = inv_d2.sum(axis=1)  # shape (n,)
    grad = sum_xj_over_d2 - pts * sum_inv_d2[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_energy, best_grad = _energy_and_grad(pts)
    best_pts = pts.copy()
    
    # Gradient descent with adaptive step size on the sphere
    lr = 0.01
    min_lr = 1e-8
    max_iter = 50000
    patience = 0
    max_patience = 200
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute gradient
        energy, grad = _energy_and_grad(pts)
        
        # Update: pts = pts - lr * grad, then project
        pts_new = pts - lr * grad
        pts_new = _project_to_sphere(pts_new)
        
        # Check for improvement
        energy_new, _ = _energy_and_grad(pts_new)
        
        if energy_new < energy - 1e-12:
            pts = pts_new
            energy = energy_new
            grad = None
            patience = 0
            # Slowly decrease learning rate
            if it > 100:
                lr *= 0.999
            if lr < min_lr:
                lr = min_lr
        else:
            patience += 1
            lr *= 0.9
            if lr < min_lr:
                lr = min_lr
            if patience > max_patience:
                break
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final polish with smaller step sizes
    for polish_lr in [1e-4, 1e-5, 1e-6, 1e-7]:
        if time.time() > deadline:
            break
        pts = best_pts.copy()
        lr = polish_lr
        for it in range(5000):
            if time.time() > deadline:
                break
            energy, grad = _energy_and_grad(pts)
            pts_new = pts - lr * grad
            pts_new = _project_to_sphere(pts_new)
            energy_new, _ = _energy_and_grad(pts_new)
            if energy_new < energy - 1e-14:
                pts = pts_new
            else:
                break
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
