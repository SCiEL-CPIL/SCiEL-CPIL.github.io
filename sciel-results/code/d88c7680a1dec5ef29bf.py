"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import math
import time


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # We only want upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clamp to avoid log of very small numbers (shouldn't happen with valid points)
    d = np.maximum(d, 1e-15)
    # Energy is sum of log(1/d) = -sum log(d)
    # Use math.fsum for precision as per task description
    log_vals = np.log(d)
    # Convert to list for math.fsum
    return -math.fsum(list(log_vals))


def _gradient_log_energy(points):
    """Compute gradient of the logarithmic energy with respect to points.
    
    Energy E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            grad[i] += diff / (d * d)
    return grad


def construct():
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimization
    x0_flat = x0.flatten()
    
    # We'll use L-BFGS-B with a penalty for sphere constraint, or use SLSQP
    # Actually, let's use a projected gradient approach or L-BFGS-B with bounds
    
    # Better approach: use L-BFGS-B but project back to sphere after each step
    # Or use scipy's minimize with SLSQP and sphere constraints
    
    # Let's use a simpler approach: iterative optimization with projection
    # Use gradient descent with projection onto sphere
    
    # Actually, let's use L-BFGS-B on the sphere by parameterizing or using penalty
    
    # Best approach: use scipy.optimize.minimize with method='L-BFGS-B' 
    # but we need to handle the sphere constraint. 
    # Alternative: use the fact that we can optimize on the sphere using 
    # Riemannian optimization, but that's complex.
    
    # Simpler: use L-BFGS-B with a large penalty for leaving the sphere
    # Or: normalize after each evaluation
    
    # Let's try: define a function that takes flattened points, normalizes each point,
    # and returns the energy. Then use L-BFGS-B.
    
    def objective(x_flat):
        pts = x_flat.reshape(N, 3)
        # Normalize each point to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        return _log_energy(pts)
    
    def gradient(x_flat):
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        g = _gradient_log_energy(pts)
        # Project gradient to tangent space: g - (g·x)x
        # For each point, the tangent space is orthogonal to the point itself
        dot = np.sum(g * pts, axis=1, keepdims=True)
        g_tangent = g - dot * pts
        return g_tangent.flatten()
    
    # Use L-BFGS-B
    result = minimize(
        objective,
        x0_flat,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
    )
    
    x_opt = result.x.reshape(N, 3)
    
    # Normalize to ensure they're on the unit sphere
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    x_opt = x_opt / norms
    
    # Verify no points are too close
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(x_opt[i] - x_opt[j])
            if d < min_dist:
                min_dist = d
    
    # If any points are too close, add small perturbation
    if min_dist < 1e-9:
        # Perturb slightly
        rng = np.random.default_rng(42)
        x_opt = x_opt + rng.normal(0, 1e-6, x_opt.shape)
        norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        x_opt = x_opt / norms
    
    return x_opt
# EVOLVE-BLOCK-END
