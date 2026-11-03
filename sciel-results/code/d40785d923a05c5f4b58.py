"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _covering_radius(pts):
    """
    Estimate the covering radius (in radians) by checking the angular distance
    from a dense set of test points on the sphere to the nearest configuration point.
    For speed, we use a Fibonacci spiral of ~20000 test points.
    """
    M = 20000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each test point to all configuration points
    # Use matrix multiplication for speed
    # dists[i, j] = 1 - dot(test_pts[i], pts[j])
    # Angular distance = arccos(1 - dists)
    # But we can just use the max of (1 - dot) as a proxy for minimizing, then convert.
    # Actually, we need the true angular distance.
    # dot product matrix
    dots = test_pts @ pts.T  # shape (M, N)
    # Clip to [-1, 1] for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Nearest point for each test point
    max_dots = np.max(dots, axis=1)
    # Angular distance
    angles = np.arccos(max_dots)
    return np.max(angles)


def _objective_and_grad(pts):
    """
    Objective: maximize the minimum pairwise angular distance (Tammes problem variant)
    or minimize the covering radius.
    
    Here we use a combined approach:
    1. Minimize the covering radius (estimated via test points)
    2. Add a penalty for points too close together (to avoid coincident points)
    
    For gradient, we use numerical differentiation if analytic is too complex.
    """
    # Normalize points
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pts_norm = pts / norms
    
    # Covering radius
    cr = _covering_radius(pts_norm)
    
    # Penalty for minimum pairwise distance
    # Compute pairwise dot products
    dots = pts_norm @ pts_norm.T
    np.fill_diagonal(dots, 1.0)
    min_dot = np.min(dots)
    min_angle = np.arccos(np.clip(min_dot, -1.0, 1.0))
    
    # We want to maximize min_angle, so minimize -min_angle
    # But also minimize covering radius
    # Combined objective: covering_radius - lambda * min_angle
    # Actually, for a good covering, we want points well-distributed.
    # Let's just minimize the covering radius directly, with a small penalty for clustering.
    
    # Penalty: if any two points are closer than some threshold, add penalty
    penalty = 0.0
    if min_angle < 0.05:  # less than ~3 degrees
        penalty = (0.05 - min_angle) ** 2 * 100
    
    return cr + penalty


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return pts / norms


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use L-BFGS-B to optimize
    # We'll flatten the points to a 1D array
    x0 = pts.flatten()
    
    def obj(x):
        pts_reshaped = x.reshape(N, 3)
        return _objective_and_grad(pts_reshaped)
    
    # Use Nelder-Mead or L-BFGS-B
    # L-BFGS-B requires gradient, so let's use Nelder-Mead which doesn't
    # But Nelder-Mead may be slow for 390 dimensions
    
    # Alternative: use a custom gradient-free optimization with better initialization
    # Or use the fact that we can compute gradients numerically
    
    # Let's try a few iterations of gradient descent with numerical gradients
    # This might be slow but should improve the Fibonacci start
    
    # Actually, let's use scipy's minimize with Nelder-Mead
    # But 390 dimensions is too many for Nelder-Mead
    
    # Better approach: use a local optimization where we move each point
    # in the direction that reduces the covering radius
    
    # For now, let's just return the Fibonacci spiral with slight improvements
    # using a few iterations of a simple optimization
    
    # Let's use a different approach: minimize the sum of inverse distances
    # or use the energy minimization approach
    
    # Energy minimization: minimize sum_{i != j} 1 / (1 - dot(pi, pj))
    # This is equivalent to minimizing the potential energy of charges on the sphere
    
    def energy(x):
        pts_reshaped = x.reshape(N, 3)
        norms = np.linalg.norm(pts_reshaped, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        pts_n = pts_reshaped / norms
        
        dots = pts_n @ pts_n.T
        np.fill_diagonal(dots, 1.0)
        # Avoid division by zero
        dots = np.clip(dots, -1.0, 0.999999)
        # Energy: sum of 1/(1-dots) for i != j
        inv_dists = 1.0 / (1.0 - dots)
        np.fill_diagonal(inv_dists, 0.0)
        return np.sum(inv_dists)
    
    # Minimize energy
    start_time = time.time()
    
    # Use L-BFGS-B with numerical gradient
    result = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 200})
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Too slow, return the Fibonacci spiral
        return pts
    
    x_opt = result.x
    pts_opt = x_opt.reshape(N, 3)
    pts_opt = _project_to_sphere(pts_opt)
    
    # Verify it's a valid configuration
    # Check no coincident points
    norms = np.linalg.norm(pts_opt, axis=1)
    if np.any(norms < 1e-6):
        return pts
    
    return pts_opt

# EVOLVE-BLOCK-END
