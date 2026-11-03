"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

def _pairwise_distances(centers):
    """Compute all pairwise distances for N points."""
    n = len(centers)
    # Compute distance matrix efficiently
    # Using broadcasting: (n, 1, 2) - (1, n, 2) -> (n, n, 2) -> norm
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    return dists

def _min_pairwise_dist(centers):
    """Minimum pairwise distance."""
    dists = _pairwise_distances(centers)
    return np.min(dists)

def _min_boundary_dist(centers):
    """Minimum distance from any center to the boundary of [-0.5, 0.5]^2."""
    # Distance to each boundary
    dist_left = centers[:, 0] + 0.5
    dist_right = 0.5 - centers[:, 0]
    dist_bottom = centers[:, 1] + 0.5
    dist_top = 0.5 - centers[:, 1]
    all_dists = np.concatenate([dist_left, dist_right, dist_bottom, dist_top])
    return np.min(all_dists)

def _radius(centers):
    """Compute the radius for a given configuration."""
    min_pd = _min_pairwise_dist(centers)
    min_bd = _min_boundary_dist(centers)
    return min(min_pd / 2.0, min_bd)

def _soft_min(values, temperature=0.01):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    # soft min = -log(sum(exp(-values/T)))*T
    min_val = np.min(values)
    exp_vals = np.exp(-(values - min_val) / temperature)
    return min_val - temperature * np.log(np.sum(exp_vals))

def _soft_radius(centers, temperature=0.01):
    """
    Smooth surrogate for the radius objective.
    We want to maximize min(min_pairwise_dist/2, min_boundary_dist).
    Equivalently, minimize -min(...).
    Using soft-min to smooth the non-smooth min.
    
    For pairwise distances, we compute all pairwise distances and take soft-min.
    For boundary distances, we compute all 4*N boundary distances and take soft-min.
    Then take soft-min of those two.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pd_vals = dists.flatten()
    
    # Boundary distances
    dist_left = centers[:, 0] + 0.5
    dist_right = 0.5 - centers[:, 0]
    dist_bottom = centers[:, 1] + 0.5
    dist_top = 0.5 - centers[:, 1]
    bd_vals = np.concatenate([dist_left, dist_right, dist_bottom, dist_top])
    
    # Soft min of all relevant distances
    all_vals = np.concatenate([pd_vals, bd_vals])
    # We want min of (pairwise/2, boundary)
    # Actually let's just minimize the negative of the soft-min of all constraints
    # The radius is min(min_pd/2, min_bd)
    # Let's compute soft-min of [all pairwise dists / 2, all boundary dists]
    pd_half = pd_vals / 2.0
    all_constraints = np.concatenate([pd_half, bd_vals])
    
    sm = _soft_min(all_constraints, temperature)
    return -sm  # We want to maximize this, so minimize the negative

def _hexagonal_lattice(n, width=1.0, height=1.0, jitter=0.0):
    """
    Generate points in a hexagonal lattice pattern within a unit square [-0.5, 0.5]^2.
    This provides a high-density initial configuration.
    """
    points = []
    # Hexagonal lattice spacing
    # For a hexagonal packing, the ratio of vertical to horizontal spacing is sqrt(3)/2
    # Let's estimate the number of rows and columns
    # Area per point ~ 1/n
    # Hexagonal cell area = (sqrt(3)/2) * s^2 where s is spacing
    # So s ~ sqrt(2/(n*sqrt(3)))
    s = np.sqrt(2.0 / (n * np.sqrt(3)))
    
    # Number of columns
    n_cols = int(np.ceil(width / s))
    # Number of rows
    row_height = s * np.sqrt(3) / 2
    n_rows = int(np.ceil(height / row_height)) + 1
    
    for row in range(n_rows):
        y = -0.5 + row * row_height
        if y > 0.5:
            break
        offset = (row % 2) * s / 2
        for col in range(n_cols):
            x = -0.5 + offset + col * s
            if x > 0.5:
                break
            points.append([x, y])
    
    points = np.array(points)
    
    # If we have too many, subsample
    if len(points) > n:
        # Randomly select n points
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Add random points
        extra = np.random.uniform(-0.5, 0.5, size=(n - len(points), 2))
        points = np.vstack([points, extra])
    
    # Apply jitter
    if jitter > 0:
        points = points + np.random.uniform(-jitter, jitter, size=points.shape)
    
    # Clip to ensure within bounds
    points = np.clip(points, -0.5, 0.5)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initial configuration: hexagonal lattice with jitter
    np.random.seed(42)
    
    # Try multiple random seeds for diversity
    n_starts = 20
    
    for seed in range(n_starts):
        if time.time() > deadline:
            break
        
        np.random.seed(seed * 12345 + 67890)
        
        # Generate initial configuration
        jitter = 0.02
        centers = _hexagonal_lattice(N, jitter=jitter)
        
        # Ensure valid initial configuration
        if not np.all(np.isfinite(centers)):
            continue
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Evaluate initial radius
        r_init = _radius(centers)
        
        # Use L-BFGS-B to optimize the soft radius
        # We minimize -soft_radius(centers)
        
        def objective(x):
            c = x.reshape(-1, 2)
            c = np.clip(c, -0.4999, 0.4999)
            return _soft_radius(c, temperature=0.01)
        
        x0 = centers.flatten()
        
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if time.time() > deadline:
                break
            
            centers_opt = result.x.reshape(-1, 2)
            centers_opt = np.clip(centers_opt, -0.4999, 0.4999)
            
            r_opt = _radius(centers_opt)
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = centers_opt.copy()
                
        except Exception:
            continue
    
    # If no valid result found, fall back to grid
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_radius = _radius(best_centers)
    
    return best_centers
# EVOLVE-BLOCK-END
