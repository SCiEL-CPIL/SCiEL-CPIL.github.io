"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_lattice_init(n):
    """Generate n points on a hexagonal lattice inside the unit disk."""
    # Estimate spacing for n points in unit disk
    # Area of unit disk = pi, area per point ~ pi/n
    # For hexagonal packing, area per point = s^2 * sqrt(3)/2 where s is spacing
    # So s ~ sqrt(2*pi / (n * sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate hexagonal lattice points
    points = []
    # Range to cover the unit disk
    k = int(1.0 / s) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) == 0:
        # Fallback: random points
        return np.random.randn(n, 2) * 0.5
    
    if len(points) < n:
        # If not enough points, add random points inside the disk
        while len(points) < n:
            x, y = np.random.randn(2)
            r = np.sqrt(x*x + y*y)
            if r <= 1.0:
                points = np.vstack([points, [[x, y]]])
        points = points[:n]
    else:
        # Select the n points closest to the center
        radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        idx = np.argsort(radii)[:n]
        points = points[idx]
    
    return points

def _compute_radius(points):
    """Compute the maximum radius for the given points."""
    n = len(points)
    if n < 2:
        return 1.0 - np.max(np.linalg.norm(points, axis=1)) if n > 0 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Minimum pairwise distance / 2
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist) / 2.0
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_pair_dist, min_boundary)

def _objective_and_grad(points):
    """Compute negative radius and its gradient (for minimization)."""
    points = np.asarray(points).reshape(-1, 2)
    n = len(points)
    r = _compute_radius(points)
    if r <= 0:
        return -r, np.zeros_like(points)
    
    # Compute gradient numerically for simplicity, or analytically
    # For now, use finite differences
    eps = 1e-7
    grad = np.zeros_like(points)
    for i in range(n):
        for d in range(2):
            p_plus = points.copy()
            p_plus[i, d] += eps
            p_minus = points.copy()
            p_minus[i, d] -= eps
            r_plus = _compute_radius(p_plus)
            r_minus = _compute_radius(p_minus)
            grad[i, d] = -(r_plus - r_minus) / (2.0 * eps)
    
    return -r, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # Initial points from hexagonal lattice
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_r = _compute_radius(best_points)
    
    # Use L-BFGS-B to optimize
    def objective(x):
        pts = x.reshape(-1, 2)
        r = _compute_radius(pts)
        return -r
    
    # Try optimization with time limit
    try:
        result = minimize(
            objective,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 10000, 'gtol': 1e-8}
        )
        
        if result.x is not None:
            new_points = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_points)
            if new_r > best_r:
                best_points = new_points
                best_r = new_r
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return best_points
    
    # Try a few more local optimizations with different initializations
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        
        # Slightly perturb the best points
        perturbed = best_points + np.random.randn(N, 2) * 0.01
        # Ensure points stay in disk
        for i in range(N):
            r = np.linalg.norm(perturbed[i])
            if r > 0.99:
                perturbed[i] = perturbed[i] * 0.99 / r
        
        try:
            result = minimize(
                objective,
                perturbed.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 5000, 'gtol': 1e-8}
            )
            
            if result.x is not None:
                new_points = result.x.reshape(-1, 2)
                new_r = _compute_radius(new_points)
                if new_r > best_r:
                    best_points = new_points
                    best_r = new_r
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
