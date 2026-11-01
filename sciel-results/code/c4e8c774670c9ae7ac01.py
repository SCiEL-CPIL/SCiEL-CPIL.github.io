"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on the unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = X.shape[0]
    # Compute all pairwise distances
    # dist[i,j] = |x_i - x_j|
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # avoid division by zero on diagonal
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of -log(dist[i,j])
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / dist[i,j]^2
    # This is the negative gradient of the energy w.r.t. x_i
    # Actually: d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2
    # So gradient of energy = sum_{j!=i} -(x_i - x_j)/dist[i,j]^2
    # Let's compute: grad[i] = sum_j (x_j - x_i)/dist[i,j]^2 for j != i
    
    # diff[i,j] = x_i - x_j
    # We want sum_j (x_j - x_i)/dist[i,j]^2 = -sum_j diff[i,j]/dist2[i,j]
    # But dist2[i,i] is set to 1, and diff[i,i]=0, so it's fine
    grad = -np.sum(diff / dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _halton_init(n, num_dims=3):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        points[i - 1] = [halton(i, 2), halton(i, 3), halton(i, 5)]
    
    # Map to sphere: use spherical coordinates
    # Or just normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    # Add some jitter to avoid exact symmetries
    rng = np.random.RandomState(42)
    points += rng.normal(0, 0.01, size=points.shape)
    return _project_to_sphere(points)


def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: kill at 150s, return by 140s
    
    # Try multiple initializations and pick the best
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_init(N))
    
    # 2. Halton sequence
    initializations.append(_halton_init(N))
    
    # 3. Random with fixed seed
    rng = np.random.RandomState(123)
    rand_pts = rng.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts))
    
    # 4. Random with different seed
    rng2 = np.random.RandomState(456)
    rand_pts2 = rng2.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts2))
    
    best_energy = np.inf
    best_points = initializations[0]
    
    for init_idx, X0 in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B with the energy function
        # We need to handle the constraint that points are on the sphere
        # Strategy: optimize in R^3 and project, or use the fact that
        # the gradient projected to the tangent space is what matters
        
        # For simplicity, we'll optimize in R^3 and project after each step
        # This is not exactly Riemannian but works well in practice
        
        def objective(x_flat):
            X = x_flat.reshape(N, 3)
            X = _project_to_sphere(X)
            e, _ = _energy_and_grad(X)
            return e
        
        def gradient(x_flat):
            X = x_flat.reshape(N, 3)
            X = _project_to_sphere(X)
            e, g = _energy_and_grad(X)
            # Project gradient to tangent space
            # tangent grad = g - (g . x_i) * x_i
            g_proj = g - np.sum(g * X, axis=1, keepdims=True) * X
            return g_proj.flatten()
        
        try:
            result = minimize(
                objective,
                X0.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            X_opt = _project_to_sphere(result.x.reshape(N, 3))
            e_opt, _ = _energy_and_grad(X_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt
        
        except Exception:
            pass
    
    # If we have time, do a high-precision polish pass on the best result
    if time.time() < deadline - 20.0:
        try:
            X0 = best_points.copy()
            
            def objective(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, _ = _energy_and_grad(X)
                return e
            
            def gradient(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, g = _energy_and_grad(X)
                g_proj = g - np.sum(g * X, axis=1, keepdims=True) * X
                return g_proj.flatten()
            
            result = minimize(
                objective,
                X0.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            X_opt = _project_to_sphere(result.x.reshape(N, 3))
            e_opt, _ = _energy_and_grad(X_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9:
        # Fall back to Fibonacci initialization
        return _fibonacci_init(N)
    
    return best_points
# EVOLVE-BLOCK-END
