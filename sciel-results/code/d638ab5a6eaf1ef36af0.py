"""Optimized logarithmic energy minimizer for 100 points on the sphere."""
import numpy as np
from scipy.optimize import minimize

N = 100


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on the unit sphere."""
    # pts: (N, 3) array of unit vectors
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid self-pairs (diagonal is 0)
    np.fill_diagonal(dist, np.inf)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum over i<j of log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    log_d = np.log(dist[iu])
    energy = -np.sum(log_d)
    
    # Gradient with respect to points
    # d/dx_i sum_{j!=i} log(1/|x_i - x_j|) = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # But we're on the sphere, so we need the tangential gradient
    # The full gradient (before projection) is:
    # g_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Project onto tangent space: g_i_tan = g_i - (g_i · x_i) x_i
    
    # Compute pairwise differences and distances
    # diff[i,j] = x_i - x_j
    # dist[i,j] = |x_i - x_j|
    
    # g_i = sum_{j!=i} (x_i - x_j) / dist[i,j]^2
    # = sum_{j!=i} diff[i,j] / dist[i,j]^2
    
    # Create mask for non-diagonal
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
    
    # g_i = sum_j (x_i - x_j) / dist[i,j]^2 for j != i
    # diff[i,j] / dist[i,j]^2
    weighted_diff = diff / dist_sq[:, :, np.newaxis]  # (N, N, 3)
    np.fill_diagonal(weighted_diff, 0.0)  # Zero out self terms
    
    grad_full = np.sum(weighted_diff, axis=1)  # (N, 3)
    
    # Project to tangent space
    dot = np.sum(grad_full * pts, axis=1)  # (N,)
    grad_tan = grad_full - dot[:, np.newaxis] * pts
    
    return energy, grad_tan


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    # Initialize with Fibonacci spiral (good starting point)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates for optimization
    # Use (theta, phi) where theta is polar angle [0, pi], phi is azimuthal [0, 2pi)
    theta0 = np.arccos(np.clip(x0[:, 2], -1.0, 1.0))
    phi0 = np.arctan2(x0[:, 1], x0[:, 0])
    
    def _spherical_to_cartesian(params):
        """Convert (theta, phi) pairs to Cartesian coordinates."""
        theta = params[0::2]  # (N,)
        phi = params[1::2]    # (N,)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    def _objective(params):
        """Objective function for L-BFGS-B."""
        pts = _spherical_to_cartesian(params)
        energy, _ = _energy_and_grad(pts)
        return energy
    
    # Optimize using L-BFGS-B with bounds
    # theta in [0, pi], phi in [0, 2pi)
    bounds = []
    for _ in range(N):
        bounds.append((0.0, np.pi))       # theta
        bounds.append((0.0, 2.0 * np.pi)) # phi
    
    # Initial parameters
    params0 = np.empty(2 * N)
    params0[0::2] = theta0
    params0[1::2] = phi0
    
    # Run optimization
    result = minimize(
        _objective,
        params0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 50000}
    )
    
    # Get optimized points
    pts = _spherical_to_cartesian(result.x)
    pts = _project_to_sphere(pts)
    
    # Verify no points are too close
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral if points are too close
        return x0
    
    return pts
