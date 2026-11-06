"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
import time

N = 100


def _energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    # Compute pairwise distances
    # points has shape (N, 3)
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Only sum upper triangle (i < j)
    iu = np.triu_indices(N, k=1)
    return np.sum(np.log(1.0 / dist[iu]))


def _energy_grad(points):
    """Compute the gradient of the logarithmic energy with respect to points on the sphere.
    
    The gradient on the ambient space is:
        grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    We then project to the tangent space of the sphere.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
    
    # grad_i = sum_j (x_i - x_j) / |x_i - x_j|^2
    # This is sum_j diff[i,j] / dist_sq[i,j]
    # diff[i,j] = x_i - x_j
    grad = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: grad_tangent = grad - (grad . x_i) x_i
    dot = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot * points
    
    return grad_tangent


def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using the golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N).copy()
    best_points = points.copy()
    best_energy = _energy(points)
    
    # L-BFGS style optimization with gradient descent
    # Use scipy if available, otherwise implement simple gradient descent
    
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        # Define objective and gradient for scipy
        def objective(x):
            pts = _project_to_sphere(x.reshape(N, 3))
            return _energy(pts)
        
        def gradient(x):
            pts = _project_to_sphere(x.reshape(N, 3))
            g = _energy_grad(pts)
            return g.flatten()
        
        # Run L-BFGS-B
        x0 = points.flatten()
        
        # Try multiple restarts if time permits
        n_restarts = 1
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                x_start = x0
            else:
                # Perturb for restart
                x_start = x0 + 0.1 * np.random.randn(N * 3)
            
            result = minimize(
                objective,
                x_start,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_energy:
                best_points = _project_to_sphere(result.x.reshape(N, 3))
                best_energy = result.fun
        
        points = best_points
    else:
        # Simple gradient descent with adaptive step size
        lr = 0.01
        for iteration in range(10000):
            if time.time() > deadline:
                break
            
            g = _energy_grad(points)
            # Adaptive learning rate
            norm_g = np.linalg.norm(g)
            if norm_g < 1e-12:
                break
            
            points = points - lr * g
            points = _project_to_sphere(points)
            
            e = _energy(points)
            if e < best_energy:
                best_energy = e
                best_points = points.copy()
            
            # Decay learning rate
            lr *= 0.999
    
    # Final polish with tighter tolerance
    points = best_points.copy()
    try:
        from scipy.optimize import minimize
        
        def objective(x):
            pts = _project_to_sphere(x.reshape(N, 3))
            return _energy(pts)
        
        def gradient(x):
            pts = _project_to_sphere(x.reshape(N, 3))
            g = _energy_grad(pts)
            return g.flatten()
        
        result = minimize(
            objective,
            points.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        if result.fun < best_energy:
            best_points = _project_to_sphere(result.x.reshape(N, 3))
            best_energy = result.fun
    except Exception:
        pass
    
    return _project_to_sphere(best_points)
