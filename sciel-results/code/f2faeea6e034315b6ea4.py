"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools


def _compute_min_triangle_area(points):
    """Compute the minimum area triangle among all triples of points."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                                  (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _compute_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _compute_min_triangle_area(points)
    return min_tri / hull_area


def _soft_min_triangle_area(points, temperature):
    """Soft minimum of triangle areas using log-sum-exp for smoothness."""
    n = points.shape[0]
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                                  (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas[i, j, k] = area
                areas[k, j, i] = area  # not used but for symmetry if needed
    
    # Only consider i < j < k
    tri_areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                                  (p3[0] - p1[0]) * (p2[1] - p1[1]))
                tri_areas.append(area)
    tri_areas = np.array(tri_areas)
    
    if len(tri_areas) == 0:
        return 0.0
    
    # Soft minimum: -log(sum(exp(-a/T))) / (n_triples) ... 
    # Actually soft min = -T * log(sum(exp(-a/T))) for T > 0
    # This gives a smooth approximation to min(a)
    scaled = -tri_areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -temperature * log_sum
    return soft_min


def _objective(points_flat, temperature):
    """Objective function to minimize: negative of soft min triangle area / hull area."""
    points = points_flat.reshape(-1, 2)
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    soft_min = _soft_min_triangle_area(points, temperature)
    return -soft_min / hull_area


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    best_points = None
    best_score = -float('inf')
    
    # Try several initial configurations
    rng = np.random.RandomState(N)
    
    # Initial guesses: regular polygon, random, etc.
    initial_configs = []
    
    # Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(regular)
    
    # Random points in unit square
    initial_configs.append(rng.rand(N, 2))
    
    # Random points in unit disk
    for _ in range(3):
        r = np.sqrt(rng.rand(N))
        theta = rng.rand(N) * 2 * np.pi
        initial_configs.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Fewer points on boundary, some interior
    for _ in range(3):
        pts = rng.rand(N, 2)
        # Push some to boundary
        for i in range(4):
            angle = rng.rand() * 2 * np.pi
            pts[i] = [np.cos(angle), np.sin(angle)]
        initial_configs.append(pts)
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        # Annealing: start with high temperature, decrease
        temperatures = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        current_points = config.copy()
        
        for temp in temperatures:
            if time.time() > deadline:
                break
            
            # Optimize from current point
            try:
                result = minimize(
                    _objective,
                    current_points.flatten(),
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.success or result.fun < 0:
                    candidate = result.x.reshape(-1, 2)
                    s = _score(candidate)
                    if s > best_score:
                        best_score = s
                        best_points = candidate.copy()
                    current_points = candidate
            except Exception:
                pass
        
        # Perturbation restarts from best found so far
        if best_points is not None:
            for _ in range(5):
                if time.time() > deadline:
                    break
                perturbed = best_points + 0.1 * rng.randn(N, 2)
                # Keep in reasonable bounds
                perturbed = np.clip(perturbed, -1, 1)
                
                for temp in [0.1, 0.05, 0.02]:
                    if time.time() > deadline:
                        break
                    try:
                        result = minimize(
                            _objective,
                            perturbed.flatten(),
                            args=(temp,),
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                        )
                        if result.success or result.fun < 0:
                            candidate = result.x.reshape(-1, 2)
                            s = _score(candidate)
                            if s > best_score:
                                best_score = s
                                best_points = candidate.copy()
                            perturbed = candidate
                    except Exception:
                        pass
    
    # If we found a valid point, return it; otherwise return a default
    if best_points is not None and _compute_hull_area(best_points) > 1e-10:
        return best_points
    
    # Fallback: regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

# EVOLVE-BLOCK-END
