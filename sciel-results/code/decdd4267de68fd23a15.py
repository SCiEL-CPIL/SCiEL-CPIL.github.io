"""Optimized 130-point sphere covering via gradient-based min-max optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 130

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms


def _covering_radius(points, n_grid=10000):
    """
    Estimate the covering radius by sampling points on the sphere.
    Returns the maximum angular distance in radians.
    """
    # Use a Fibonacci spiral for dense sampling
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each grid point to all configuration points
    # Use dot products for efficiency
    dots = grid @ points.T  # shape (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angular distance = arccos(dot)
    # We want the max over grid of min over points of arccos(dot)
    # Equivalent to max over grid of arccos(max over points of dot)
    max_dots = np.max(dots, axis=1)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    
    return np.max(angles)


def _smooth_coverage_field(points, grid, sigma):
    """
    Compute a smooth coverage field using Gaussian RBF.
    Higher values indicate better coverage.
    We minimize the max of (1 - coverage) or equivalently maximize the min coverage.
    """
    # dots: (n_grid, N)
    dots = grid @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # Gaussian RBF: exp(-d^2 / (2*sigma^2)) where d is angular distance
    # Use exp(-(1-dot)^2 / (2*sigma^2)) as a proxy for angular distance squared
    # Actually, angular distance d satisfies cos(d) = dot, so d^2 ≈ 2(1-dot) for small d
    # Use exp(-(1-dot)/(sigma^2)) for a smoother field
    field = np.exp(-(1.0 - dots) / (sigma * sigma))
    
    # Return the minimum field value (worst-covered point)
    # We want to maximize this, so we minimize its negative
    return -np.min(field)


def _objective_and_grad(x, grid, sigma, use_smooth=True):
    """
    Objective function and gradient for optimization.
    x: flattened array of N points (3N values)
    """
    points = x.reshape(N, 3)
    points = _normalize(points)
    
    if use_smooth:
        # Smooth objective: minimize -min(exp(-(1-dot)/sigma^2))
        dots = grid @ points.T  # (n_grid, N)
        dots = np.clip(dots, -1.0, 1.0)
        
        field = np.exp(-(1.0 - dots) / (sigma * sigma))
        min_val = np.min(field)
        
        # Find which grid point has the minimum field
        min_idx = np.argmin(field)
        
        # Gradient with respect to points
        # d/dp of -min(field) = -d/dp(field[min_idx])
        # field[i] = exp(-(1 - dots[i,j])/sigma^2) where j is the point that minimizes
        # Actually, for each grid point i, field[i] = max_j exp(-(1-dots[i,j])/sigma^2) is not right
        # Let me reconsider: we want to maximize min_i max_j f(i,j)
        # where f(i,j) = exp(-(1 - dots[i,j])/sigma^2)
        
        # This is complex. Let's use a simpler approach:
        # Minimize the max angular distance using a soft-min
        
        # Actually, let's use the direct approach with subgradient
        # For now, use numerical gradient or a simpler smooth approximation
        
        # Alternative: minimize sum of exp(beta * max_angle_i) where max_angle_i is 
        # the max angular distance from grid point i to nearest point
        # This is a log-sum-exp type smooth max
        
        beta = 50.0  # temperature parameter
        
        # Compute max angular distance for each grid point
        max_dots = np.max(dots, axis=1)
        max_dots = np.clip(max_dots, -1.0, 1.0)
        angles = np.arccos(max_dots)
        
        # Smooth max: (1/beta) * log(sum(exp(beta * angles)))
        # For stability, subtract max
        angles_max = np.max(angles)
        smooth_max = angles_max + np.log(np.sum(np.exp(beta * (angles - angles_max)))) / beta
        
        # Gradient: d(smooth_max)/d(points)
        # This requires chain rule through arccos and max, which is tricky
        # Use finite differences for gradient
        
        return smooth_max, None  # Use numerical gradient
    
    else:
        # Direct covering radius
        cr = _covering_radius(points, n_grid=5000)
        return cr, None


def _gradient_descent(points, max_iter=200, lr=0.01, tol=1e-6, deadline=None):
    """
    Simple gradient descent on the sphere using tangent space projection.
    """
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(best_points, n_grid=5000)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Compute gradient numerically
        grad = np.zeros_like(points)
        eps = 1e-5
        base_cr = _covering_radius(points, n_grid=3000)
        
        for i in range(N):
            for j in range(3):
                perturbed = points.copy()
                perturbed[i, j] += eps
                perturbed[i] = perturbed[i] / np.linalg.norm(perturbed[i])
                cr_plus = _covering_radius(perturbed, n_grid=3000)
                
                perturbed2 = points.copy()
                perturbed2[i, j] -= eps
                perturbed2[i] = perturbed2[i] / np.linalg.norm(perturbed2[i])
                cr_minus = _covering_radius(perturbed2, n_grid=3000)
                
                grad[i, j] = (cr_plus - cr_minus) / (2 * eps)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Update
        points = points - lr * grad_tangent
        points = _normalize(points)
        
        # Check improvement
        cr = _covering_radius(points, n_grid=3000)
        if cr < best_cr - 1e-7:
            best_cr = cr
            best_points = points.copy()
        else:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-6:
                break
    
    return best_points, best_cr


def _lloyd_iteration(points, max_iter=50, deadline=None):
    """
    Lloyd-like iteration: for each point, move it to the centroid of its 
    Voronoi cell (approximated by nearest grid points).
    """
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(best_points, n_grid=5000)
    
    # Create a dense grid
    n_grid = 10000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Assign each grid point to nearest configuration point
        dots = grid @ points.T  # (n_grid, N)
        nearest = np.argmax(dots, axis=1)
        
        # Compute new positions as centroids of assigned grid points
        new_points = np.zeros_like(points)
        for k in range(N):
            mask = nearest == k
            if np.sum(mask) > 0:
                centroid = np.mean(grid[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_points[k] = centroid / norm
                else:
                    new_points[k] = points[k]
            else:
                new_points[k] = points[k]
        
        new_points = _normalize(new_points)
        
        # Blend with old points for stability
        alpha = 0.5
        points = (1 - alpha) * points + alpha * new_points
        points = _normalize(points)
        
        cr = _covering_radius(points, n_grid=5000)
        if cr < best_cr - 1e-7:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _refine_with_scipy(points, deadline):
    """
    Use scipy.optimize.minimize with BFGS for final refinement.
    """
    if deadline is not None and time.time() > deadline - 10:
        return points
    
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius(pts, n_grid=5000)
    
    try:
        result = minimize(objective, x0, method='BFGS', 
                         options={'maxiter': 50, 'gtol': 1e-5},
                         timeout=max(0, deadline - time.time() - 5) if deadline else None)
        if result.success or result.fun < _covering_radius(points, n_grid=5000):
            return _normalize(result.x.reshape(N, 3))
    except Exception:
        pass
    
    return points


def construct():
    """Construct 130 points on unit sphere minimizing covering radius."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    points = _normalize(points)
    
    best_points = points.copy()
    best_cr = _covering_radius(best_points, n_grid=10000)
    
    # Phase 1: Lloyd iterations
    points, cr = _lloyd_iteration(points, max_iter=30, deadline=deadline)
    if cr < best_cr:
        best_cr = cr
        best_points = points.copy()
    
    # Phase 2: Gradient descent
    if time.time() < deadline - 20:
        points, cr = _gradient_descent(best_points, max_iter=100, lr=0.05, deadline=deadline)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 3: More Lloyd iterations
    if time.time() < deadline - 15:
        points, cr = _lloyd_iteration(best_points, max_iter=30, deadline=deadline)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 4: Final refinement with scipy
    if time.time() < deadline - 10:
        best_points = _refine_with_scipy(best_points, deadline)
        final_cr = _covering_radius(best_points, n_grid=10000)
    
    return _normalize(best_points)

# EVOLVE-BLOCK-END
