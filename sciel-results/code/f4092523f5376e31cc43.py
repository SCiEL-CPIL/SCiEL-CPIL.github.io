# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses a multi-start optimization strategy:
    1. Generate diverse initial configurations (random, spherical, etc.)
    2. For each, run Nelder-Mead optimization on the objective
    3. Return the best configuration found
    
    The objective is to maximize (d_min/d_max)^2, equivalently minimize -(d_min/d_max)^2.
    We use a smooth surrogate to handle the non-differentiable min/max.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, use 630s wall clock with margin
    safety_margin = 30.0  # seconds before deadline to stop
    
    n = 14
    d = 3
    
    def compute_metric(points):
        """Compute (d_min/d_max)^2 for a set of points."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        if d_min == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def objective(points):
        """Objective to minimize: negative of (d_min/d_max)^2 with smoothing."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        if d_min < 1e-15:
            return 0.0
        ratio = d_min / d_max
        return -ratio ** 2
    
    def objective_smooth(points, alpha=50.0):
        """Smooth surrogate using soft-min and soft-max via log-sum-exp."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        
        # Soft min: -log(sum(exp(-alpha * dists))) / alpha
        # Soft max: log(sum(exp(alpha * dists))) / alpha
        neg_dists = -alpha * dists
        # Clip to avoid overflow
        neg_dists_clipped = np.clip(neg_dists, -700, 700)
        soft_min = -np.log(np.sum(np.exp(neg_dists_clipped))) / alpha
        
        pos_dists = alpha * dists
        pos_dists_clipped = np.clip(pos_dists, -700, 700)
        soft_max = np.log(np.sum(np.exp(pos_dists_clipped))) / alpha
        
        if soft_max < 1e-15:
            return 0.0
        if soft_min < 1e-15:
            return 0.0
        
        ratio = soft_min / soft_max
        return -ratio ** 2
    
    def optimize_from_start(start_points, max_iter=2000, tol=1e-10):
        """Run Nelder-Mead optimization from a starting point."""
        try:
            res = minimize(
                objective,
                start_points,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': tol,
                    'fatol': tol,
                    'adaptive': True
                }
            )
            return res.x, compute_metric(res.x)
        except Exception:
            return start_points, compute_metric(start_points)
    
    def generate_random_start(seed=None):
        """Generate a random starting configuration."""
        if seed is not None:
            np.random.seed(seed)
        points = np.random.randn(n, d)
        # Center and scale
        points -= np.mean(points, axis=0)
        norm = np.linalg.norm(points)
        if norm > 0:
            points /= norm
        return points
    
    def generate_spherical_start(seed=None):
        """Generate points roughly on a sphere."""
        if seed is not None:
            np.random.seed(seed)
        points = np.random.randn(n, d)
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / (norms + 1e-10)
        return points
    
    def generate_tetrahedral_start():
        """Generate a symmetric starting configuration based on tetrahedral symmetry."""
        # Start with regular tetrahedron vertices scaled
        # Tetrahedron vertices in 3D
        tetra = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=float)
        tetra = tetra / np.linalg.norm(tetra[0])
        
        # Create 14 points: 4 vertices + 10 edge midpoints
        points = [tetra[0], tetra[1], tetra[2], tetra[3]]
        # Edge midpoints
        for i in range(4):
            for j in range(i+1, 4):
                mid = (tetra[i] + tetra[j]) / 2
                points.append(mid)
        
        points = np.array(points)
        # Center
        points -= np.mean(points, axis=0)
        # Scale to unit norm
        norm = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if norm > 0:
            points /= norm
        return points
    
    def generate_fibonacci_sphere_start():
        """Generate points using Fibonacci sphere distribution."""
        n_pts = n
        golden_angle = np.pi * (3 - np.sqrt(5))
        points = np.zeros((n_pts, d))
        for i in range(n_pts):
            y = 1 - (i / (n_pts - 1)) * 2
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            points[i, 0] = np.cos(theta) * radius
            points[i, 1] = y
            points[i, 2] = np.sin(theta) * radius
        return points
    
    # Generate multiple starting configurations
    starts = []
    
    # Symmetric starts
    starts.append(generate_tetrahedral_start())
    starts.append(generate_fibonacci_sphere_start())
    
    # Random starts with different seeds
    for seed in range(20):
        starts.append(generate_random_start(seed))
        starts.append(generate_spherical_start(seed))
    
    # Optimize from each start
    best_points = None
    best_metric = 0.0
    
    for idx, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - safety_margin:
            break
        
        opt_points, metric = optimize_from_start(start, max_iter=1500, tol=1e-10)
        
        if metric > best_metric:
            best_metric = metric
            best_points = opt_points.copy()
    
    # If we still have time, try to polish the best solution
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < deadline - safety_margin:
        # Do a final polish with tighter tolerance
        try:
            res = minimize(
                objective,
                best_points,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            polished_metric = compute_metric(res.x)
            if polished_metric > best_metric:
                best_points = res.x
                best_metric = polished_metric
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to random
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    return best_points

# EVOLVE-BLOCK-END
