"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute pairwise distances
    # Use broadcasting: dist[i,j] = |points[i] - points[j]|
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid division by zero / log(0) for i==j (dist=0)
    # We only sum over i<j, so we can mask the diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    # Filter out any distances that are too small (shouldn't happen if points are distinct)
    # But for safety, clip to a small positive value
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return math.fsum(terms)


def _log_energy_grad(points):
    """Compute the gradient of the logarithmic energy with respect to points."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # d/dp_i log(1/d) = -1/d * d/dp_i d = -1/d * diff/d = -diff/d^2
            # d/dp_j log(1/d) = -1/d * d/dp_j d = -1/d * (-diff)/d = diff/d^2
            factor = 1.0 / (d * d)
            grad[i] -= diff * factor
            grad[j] += diff * factor
    return grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def construct():
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to unit sphere
    points = _project_to_sphere(points)
    
    # Use L-BFGS-B to minimize the energy
    # We need to handle the constraint that points lie on the sphere.
    # One approach: parameterize points in R^3 and use a penalty or projection.
    # Better: use the fact that the gradient on the sphere is the tangential component.
    # grad_sphere_i = grad_i - (grad_i . x_i) * x_i
    
    def energy_and_grad(x):
        pts = x.reshape(N, 3)
        # Project to sphere to ensure we're on the manifold
        pts = _project_to_sphere(pts)
        e = _log_energy(pts)
        g = _log_energy_grad(pts)
        # Project gradient to tangent space
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_tan.reshape(-1)
    
    # Initial point
    x0 = points.reshape(-1)
    
    # Run optimization
    res = minimize(
        fun=lambda x: energy_and_grad(x)[0],
        x0=x0,
        jac=lambda x: energy_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-15}
    )
    
    final_points = _project_to_sphere(res.x.reshape(N, 3))
    
    # Verify all points are on the sphere and distinct
    norms = np.linalg.norm(final_points, axis=1)
    if np.any(norms < 1e-15) or np.any(~np.isfinite(final_points)):
        # Fallback to initial points
        return points
    
    # Check for very close points
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        # Fallback to initial points
        return points
    
    return final_points
# EVOLVE-BLOCK-END
