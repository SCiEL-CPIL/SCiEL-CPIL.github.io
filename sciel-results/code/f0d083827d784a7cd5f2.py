"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy_and_grad(pts):
    # pts: (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 1.0)
    # Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Gradient: d/dx_i sum_{j!=i} log(1/|x_i - x_j|) = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # But we want to minimize, so gradient of energy w.r.t. x_i = -sum_{j!=i} (x_i - x_j)/|x_i-x_j|^2
    # Wait: E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j != i} d/dx_i log(|x_i - x_j|) = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-15
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    # grad[i] = -sum_j (x_i - x_j) / |x_i - x_j|^2 for j != i
    # diff[i,j] = x_i - x_j
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Compute energy
    energy = -np.sum(np.log(dists[np.triu_indices(N, k=1)]))
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initial point: Fibonacci sphere
    x0 = _fibonacci_sphere()
    e0, _ = _log_energy_and_grad(x0)
    best_pts = x0.copy()
    best_energy = e0
    
    # Use L-BFGS-B with custom gradient
    # We optimize on R^3 and project back to sphere after each iteration
    # Actually, better to use a parameterization or just do projected gradient
    
    # Simple approach: use scipy minimize with L-BFGS-B, 
    # but we need to handle the constraint that points are on unit sphere.
    # Alternative: use a local optimization where we move points on the sphere.
    
    # Let's use a simple gradient descent on the sphere with adaptive step size
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        e, g = _log_energy_and_grad(best_pts)
        if e < best_energy:
            best_energy = e
        
        # Project gradient to tangent space
        # Tangent projection: g_tangent = g - (g . x) * x
        g_tangent = g - np.sum(g * best_pts, axis=1, keepdims=True) * best_pts
        
        # Update
        new_pts = best_pts - lr * g_tangent
        best_pts = _project_to_sphere(new_pts)
        
        # Adaptive learning rate
        e_new, _ = _log_energy_and_grad(best_pts)
        if e_new > e:
            lr *= 0.5
            # Revert
            best_pts = best_pts + lr * g_tangent  # This is approximate, let's just keep old
            best_pts = _project_to_sphere(best_pts + lr * g_tangent)
        else:
            if iteration % 100 == 0 and lr < 0.01:
                lr *= 1.1
    
    # Final polish with smaller steps
    lr = 0.001
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        e, g = _log_energy_and_grad(best_pts)
        if e < best_energy:
            best_energy = e
        
        g_tangent = g - np.sum(g * best_pts, axis=1, keepdims=True) * best_pts
        new_pts = _project_to_sphere(best_pts - lr * g_tangent)
        
        e_new, _ = _log_energy_and_grad(new_pts)
        if e_new < e:
            best_pts = new_pts
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_pts
# EVOLVE-BLOCK-END
