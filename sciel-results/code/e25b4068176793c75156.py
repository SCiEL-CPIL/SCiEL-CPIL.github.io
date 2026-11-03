import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad_sphere(pts):
    """
    Compute log energy and gradient for points on unit sphere.
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient w.r.t pts[i] (tangent space)
    """
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid div by zero
    
    dist = np.sqrt(dist_sq)
    dist[dist < 1e-15] = 1e-15
    
    # Energy
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient
    # grad[i] = -sum_{j!=i} (x_i - x_j) / dist[i,j]^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space (remove radial component)
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial * pts
    
    return energy, grad

def _objective_ambient(x):
    """
    Objective function and gradient for L-BFGS-B on ambient space.
    x is shape (N*3,)
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    pts_norm = pts / norms
    
    e, g_sphere = _energy_and_grad_sphere(pts_norm)
    
    # Gradient w.r.t ambient x
    # dE/dx_i = (1/||x_i||) * (I - x_i x_i^T / ||x_i||^2) * dE/dp_i
    # Since dE/dp_i is tangent, (I - ...) dE/dp_i = dE/dp_i
    # So dE/dx_i = dE/dp_i / ||x_i||
    g_ambient = g_sphere / norms
    
    return e, g_ambient.flatten()

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci
    inits.append(_fibonacci_sphere(N).flatten())
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(_random_sphere(N, rng).flatten())
        
    # 3. Perturbed Fibonacci
    for _ in range(5):
        pts = _fibonacci_sphere(N) + 0.1 * rng.standard_normal((N, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        inits.append(pts.flatten())

    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                _objective_ambient, 
                init, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            
            if res.success:
                pts = res.x.reshape(N, 3)
                # Project to sphere to be safe
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1e-12
                pts = pts / norms
                
                e, _ = _energy_and_grad_sphere(pts)
                if e < best_energy:
                    best_energy = e
                    best_pts = pts.copy()
                    
        except Exception:
            pass

    # Final polish on the best result
    if best_pts is not None:
        # Run a few more iterations of L-BFGS-B from the best point
        try:
            res = minimize(
                _objective_ambient, 
                best_pts.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1e-12
            pts = pts / norms
            
            e, _ = _energy_and_grad_sphere(pts)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
        except Exception:
            pass

    return best_pts
