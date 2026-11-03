"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum


def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = len(pts)
    # Pairwise distances
    # Use broadcasting for speed
    # dist[i,j] = |xi - xj|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # We want sum over i<j of log(1/dist) = -sum over i<j log(dist)
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    # Ensure no zeros (shouldn't happen after fill_diagonal, but just in case)
    d_upper = np.maximum(d_upper, 1e-15)
    # Use fsum for precision as required by task description?
    # The task says: "pair terms summed with math.fsum"
    # Let's compute the list of values and use fsum
    vals = -np.log(d_upper).tolist()
    return fsum(vals)


def _grad(pts):
    """Compute gradient of energy w.r.t. points.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    d_ij = |xi - xj|
    dE/dxi = sum_{j!=i} (xi - xj) / |xi - xj|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    # diff[i,j] = xi - xj
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dist_sq[i,j] = |xi - xj|^2
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    # inv_dist_sq
    inv_dist_sq = 1.0 / dist_sq
    # grad_i = sum_j diff[i,j] * inv_dist_sq[i,j]
    # Note: diagonal term is 0 because diff[i,i]=0, so it's fine.
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _barzilai_borwein(pts, grad_prev, step_prev, alpha):
    """Compute Barzilai-Borwein step size."""
    # s = x_k - x_{k-1}
    # y = grad_k - grad_{k-1}
    # alpha = (s^T s) / (s^T y) for BB1 or (s^T y) / (y^T y) for BB2
    # We need previous gradient and previous point.
    # This is a bit complex to manage state.
    # Let's just use a simple adaptive step size or fixed small step.
    # Given time constraints, a simple gradient descent with adaptive step might be safer.
    pass


def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _energy(best_pts)
    
    # Gradient Descent with adaptive step size
    # We'll use a simple line search or fixed step with decay.
    # Since we have a time limit, we'll do a fixed number of iterations or until time runs out.
    
    # Let's use a simple gradient descent with backtracking line search.
    step_size = 0.01
    lr = step_size
    prev_grad = None
    
    for it in range(100000):
        if time.time() > deadline:
            break
        
        grad = _grad(pts)
        
        # Project gradient to tangent space: grad - (grad . x) x
        # For unit sphere constraint, the gradient of the energy function restricted to the sphere
        # is the projection of the Euclidean gradient onto the tangent space.
        # grad_tangent = grad - (grad * pts) * pts
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Normalize gradient for step size control
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
            
        # Backtracking line search
        # f(pts + alpha * grad_tangent) projected back to sphere
        # We want to minimize E
        # Let's try a few step sizes
        
        alpha = lr
        current_energy = best_energy if it == 0 else _energy(pts)
        
        improved = False
        for _ in range(10):
            new_pts = pts + alpha * grad_tangent
            new_pts = _project_to_sphere(new_pts)
            new_energy = _energy(new_pts)
            if new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            # If no improvement, reduce learning rate
            lr *= 0.5
            if lr < 1e-12:
                break
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = pts.copy()
    
    # Final polish: high precision local solve
    # Use L-BFGS-B from scipy.optimize for a final polish
    from scipy.optimize import minimize
    
    def fun(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        return _energy(p)
    
    def jac(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        g = _grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    x0 = best_pts.flatten()
    
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10})
        if res.success and res.fun < best_energy:
            best_pts = _project_to_sphere(res.x.reshape(-1, 3))
            best_energy = res.fun
    except Exception:
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
