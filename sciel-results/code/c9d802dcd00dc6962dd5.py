"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import numpy as np
from scipy.optimize import minimize
import time

N = 150


def _energy_and_grad(x):
    """
    Compute the logarithmic energy and its gradient for points on the unit sphere.
    x: shape (N, 3)
    Returns: energy (scalar), gradient (N, 3)
    """
    # Compute pairwise distances
    # dist[i,j] = |x_i - x_j|
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    log_d = np.log(dist[iu])
    energy = -np.sum(log_d)
    
    # Gradient: d/dx_i [ -log(|x_i - x_j|) ] = -(x_i - x_j) / |x_i - x_j|^2
    # For each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: we want to minimize E = sum log(1/d) = -sum log(d)
    # dE/dx_i = -d/dx_i [sum_{j!=i} log(d_ij)]
    # d/dx_i log(d_ij) = (x_i - x_j) / d_ij^2
    # So dE/dx_i = -sum_{j!=i} (x_i - x_j) / d_ij^2 = sum_{j!=i} (x_j - x_i) / d_ij^2
    
    # Vectorized gradient computation
    # For pair (i,j), contribution to grad_i is (x_j - x_i)/d_ij^2
    # and contribution to grad_j is (x_i - x_j)/d_ij^2
    
    # inv_d2 = 1/d_ij^2 for i != j
    inv_d2 = np.zeros((N, N))
    mask = ~np.eye(N, dtype=bool)
    inv_d2[mask] = 1.0 / (dist[mask] ** 2)
    
    # grad_i = sum_j (x_j - x_i) * inv_d2[i,j]
    # = sum_j x_j * inv_d2[i,j] - x_i * sum_j inv_d2[i,j]
    sum_inv_d2 = np.sum(inv_d2, axis=1)  # (N,)
    # sum_j x_j * inv_d2[i,j] for each i
    weighted_sum = inv_d2 @ x  # (N, 3)
    grad = weighted_sum - x * sum_inv_d2[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(x):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return x / norms


def _halton_init():
    """Generate a good initial configuration using Halton sequences."""
    def halton(index, base):
        """Generate the Halton sequence value."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        # Use three different bases for 3D
        u = halton(i, 2)
        v = halton(i, 3)
        w = halton(i, 5)
        
        # Map to sphere using a simple method
        # Use the fact that uniform points in [0,1]^3 can be mapped to sphere
        # via a transformation, but simpler: use Fibonacci-like with Halton
        z = 2.0 * w - 1.0
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = 2.0 * np.pi * u
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x, y, z])
    
    return np.array(points)


def _fibonacci_init():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """
    Minimize the logarithmic energy of 150 points on the unit sphere.
    Returns an array of shape (N, 3) with points on the unit sphere.
    """
    deadline = time.time() + 110.0  # Leave 10s safety margin
    
    # Try multiple initializations and pick the best
    initializations = [
        _fibonacci_init(),
        _halton_init(),
    ]
    
    best_points = None
    best_energy = np.inf
    
    for init_points in initializations:
        if time.time() > deadline:
            break
        
        # Flatten for scipy.optimize
        x0 = init_points.flatten()
        
        def objective_and_grad(x):
            pts = x.reshape(N, 3)
            pts = _project_to_sphere(pts)
            e, g = _energy_and_grad(pts)
            return e, g
        
        # Use BFGS with custom line search
        try:
            res = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='BFGS',
                options={
                    'maxiter': 500,
                    'gtol': 1e-10,
                    'disp': False,
                }
            )
            
            if res.success or res.fun < best_energy:
                best_points = _project_to_sphere(res.x.reshape(N, 3))
                best_energy = res.fun
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # If no successful optimization, return the best initialization
    if best_points is None:
        # Evaluate all initializations
        for init_points in initializations:
            e, _ = _energy_and_grad(init_points)
            if e < best_energy:
                best_energy = e
                best_points = init_points
    
    # Final polish: do a few more iterations with tighter tolerance if time allows
    if time.time() < deadline - 5.0 and best_points is not None:
        x0 = best_points.flatten()
        
        def objective_and_grad(x):
            pts = x.reshape(N, 3)
            pts = _project_to_sphere(pts)
            e, g = _energy_and_grad(pts)
            return e, g
        
        try:
            res = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='BFGS',
                options={
                    'maxiter': 200,
                    'gtol': 1e-12,
                    'disp': False,
                }
            )
            
            if res.fun < best_energy:
                best_points = _project_to_sphere(res.x.reshape(N, 3))
                best_energy = res.fun
        except Exception:
            pass
    
    # Ensure points are on the unit sphere
    if best_points is not None:
        best_points = _project_to_sphere(best_points)
        # Verify no points are too close
        norms = np.linalg.norm(best_points, axis=1)
        if np.any(norms < 1e-15) or not np.all(np.isfinite(best_points)):
            # Fallback to Fibonacci
            best_points = _fibonacci_init()
    
    return best_points
