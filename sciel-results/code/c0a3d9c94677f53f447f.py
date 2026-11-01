"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time

N = 150

def _energy_and_grad(X):
    """Compute logarithmic energy and Riemannian gradient for points on the unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Riemannian Gradient w.r.t. x_i = (I - x_i x_i^T) * (sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2)
    """
    n = X.shape[0]
    # Compute all pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # avoid division by zero on diagonal
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Euclidean Gradient: grad_eucl[i] = sum_{j!=i} (x_j - x_i) / dist[i,j]^2
    # Note: diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    # grad_eucl[i] = - sum_j diff[i,j] / dist2[i,j]
    eucl_grad = -np.sum(diff / dist2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: grad_riem[i] = eucl_grad[i] - (eucl_grad[i] . X[i]) X[i]
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X
    
    return energy, riem_grad

def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

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
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    # Add some jitter
    rng = np.random.RandomState(42)
    points += rng.normal(0, 0.01, size=points.shape)
    return _project_to_sphere(points)

def _riemannian_gd(X0, max_iter=1000, tol=1e-10, time_limit=10.0):
    """Riemannian Gradient Descent with backtracking line search."""
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < tol:
        return X, energy
    
    alpha = 0.1  # initial step size
    c = 1e-4     # Armijo parameter
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Backtracking line search
        step = alpha * grad
        X_new = X - step
        X_new = _project_to_sphere(X_new)
        
        energy_new, _ = _energy_and_grad(X_new)
        
        # Armijo condition: f(x_new) <= f(x) - c * alpha * ||grad||^2
        while energy_new > energy - c * alpha * grad_norm**2:
            alpha *= 0.5
            step = alpha * grad
            X_new = X - step
            X_new = _project_to_sphere(X_new)
            energy_new, _ = _energy_and_grad(X_new)
            if alpha < 1e-15:
                break
        
        if alpha < 1e-15:
            break
            
        X = X_new
        energy = energy_new
        _, grad = _energy_and_grad(X)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
            
    return X, energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
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
    
    # 5. Another random seed
    rng3 = np.random.RandomState(789)
    rand_pts3 = rng3.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts3))
    
    best_energy = np.inf
    best_points = initializations[0]
    
    # Phase 1: Riemannian GD from multiple starts
    for init_idx, X0 in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
            
        # Use a time limit for each run to ensure we finish
        time_budget = (deadline - 30.0 - time.time()) / max(1, len(initializations) - init_idx)
        time_budget = max(time_budget, 5.0)
        
        try:
            X_opt, e_opt = _riemannian_gd(X0, max_iter=500, tol=1e-8, time_limit=time_budget)
            
            # Polish with L-BFGS-B for better local convergence
            def objective(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, _ = _energy_and_grad(X)
                return e
            
            def gradient(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, g = _energy_and_grad(X)
                return g.flatten()
                
            # Only polish if we have time
            if time.time() < deadline - 20.0:
                result = minimize(
                    objective,
                    X_opt.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                X_polished = _project_to_sphere(result.x.reshape(N, 3))
                e_polished, _ = _energy_and_grad(X_polished)
                
                if e_polished < e_opt:
                    X_opt = X_polished
                    e_opt = e_polished
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt
                
        except Exception:
            pass

    # Phase 2: Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
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
                return g.flatten()
                
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
