"""Optimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """Compute log energy and gradient for points on unit sphere (chordal distances).
    
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    where d_ij = |x_i - x_j| (chordal distance)
    
    Gradient of -log(d_ij) w.r.t. x_i:
      d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j) / |x_i - x_j|^2
    """
    # Pairwise differences: (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    dist[dist < 1e-15] = 1e-15
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j!=i} -(x_i - x_j)/|x_i - x_j|^2
    # = sum_{j!=i} (x_j - x_i)/|x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so (x_j - x_i) = -diff[i,j]
    # grad_i = sum_j -diff[i,j] / dist[i,j]^2  for j != i
    #         = -sum_j diff[i,j] / dist[i,j]^2  (diagonal is 0 since diff[i,i]=0)
    
    # diff / dist^2: (N, N, 3)
    inv_dist_sq = 1.0 / (dist * dist)  # (N, N)
    inv_dist_sq[~np.isfinite(inv_dist_sq)] = 0.0
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff[i,j] = x_i - x_j
    # So grad[i] = -sum_j (x_i - x_j) * inv_dist_sq[i,j] = sum_j (x_j - x_i) * inv_dist_sq[i,j]
    # = sum_j (x_j - x_i) / dist[i,j]^2
    
    # Compute: grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff shape (N,N,3), inv_dist_sq shape (N,N)
    # grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project gradient to tangent space of sphere: remove radial component
    # grad_tangent = grad - (grad . x_i) * x_i
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial * pts
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1e-15
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points as initial guess."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_pts = _fibonacci_sphere(N)
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Try multiple initializations
    inits = [best_pts.copy()]
    
    # Random initializations
    for _ in range(5):
        if time.time() > deadline - 60:
            break
        rng = np.random.default_rng()
        pts = rng.standard_normal((N, 3))
        pts = _project_to_sphere(pts)
        inits.append(pts)
    
    # Also try perturbed Fibonacci
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        rng = np.random.default_rng()
        pts = _fibonacci_sphere(N) + 0.1 * rng.standard_normal((N, 3))
        pts = _project_to_sphere(pts)
        inits.append(pts)
    
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline - 30:
            break
        
        # Use L-BFGS-B with the energy function
        def obj_and_grad(x):
            pts = _project_to_sphere(x)
            e, g = _energy_and_grad(pts)
            # Project gradient back to ambient space for unconstrained optimizer
            # Actually, we should work on the sphere directly.
            # Let's use a different approach: parameterize via ambient coordinates
            # and project after each step.
            return e, g
        
        # We'll do gradient descent on the sphere directly with adaptive step size
        pts = init.copy()
        current_energy = _energy_and_grad(pts)[0]
        
        # Gradient descent with backtracking line search on the sphere
        lr = 0.01
        max_iter = 2000
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            e, g = _energy_and_grad(pts)
            current_energy = e
            
            # Check convergence
            if np.max(np.abs(g)) < 1e-10:
                break
            
            # Gradient step
            new_pts = pts - lr * g
            new_pts = _project_to_sphere(new_pts)
            
            new_e = _energy_and_grad(new_pts)[0]
            
            # Backtracking line search
            if not np.isfinite(new_e) or new_e > current_energy:
                lr *= 0.5
                if lr < 1e-12:
                    break
                continue
            
            pts = new_pts
            # Adaptive learning rate
            if new_e < current_energy * 0.999:
                lr = min(lr * 1.1, 0.1)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = pts.copy()
    
    # Final polish with smaller learning rate
    pts = best_pts.copy()
    for lr in [0.001, 0.0001, 0.00001]:
        for it in range(500):
            if time.time() > deadline - 5:
                break
            e, g = _energy_and_grad(pts)
            if np.max(np.abs(g)) < 1e-12:
                break
            new_pts = _project_to_sphere(pts - lr * g)
            new_e = _energy_and_grad(new_pts)[0]
            if not np.isfinite(new_e) or new_e >= e:
                continue
            pts = new_pts
        if time.time() > deadline - 5:
            break
    
    if _energy_and_grad(pts)[0] < best_energy:
        best_pts = pts
    
    return best_pts
