# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses a hexagonal lattice seed with gradient-based local optimization.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin
    
    n = 13
    
    def compute_min_triangle_area(points):
        """Compute the minimum area among all triangles formed by 3 points."""
        min_area = np.inf
        for i, j, k in combinations(range(len(points)), 3):
            x1, y1 = points[i]
            x2, y2 = points[j]
            x3, y3 = points[k]
            area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
            if area < min_area:
                min_area = area
        return min_area
    
    def compute_hull_area(points):
        """Compute the area of the convex hull."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0
    
    def score(points):
        """Compute the score: A_min / A_hull."""
        a_min = compute_min_triangle_area(points)
        a_hull = compute_hull_area(points)
        if a_hull <= 0 or not np.isfinite(a_min):
            return 0.0
        return a_min / a_hull
    
    def negative_score(points_flat):
        """Negative score for minimization."""
        points = points_flat.reshape(-1, 2)
        s = score(points)
        return -s
    
    # Create a hexagonal lattice configuration for 13 points
    # Center + 6 inner ring + 6 outer ring
    def make_hexagonal_seed():
        points = []
        # Center
        points.append([0.0, 0.0])
        
        # Inner ring (6 points at radius 1)
        for i in range(6):
            angle = i * np.pi / 3
            points.append([np.cos(angle), np.sin(angle)])
        
        # Outer ring (6 points at radius 2, offset by 30 degrees)
        for i in range(6):
            angle = (i + 0.5) * np.pi / 3
            points.append([2 * np.cos(angle), 2 * np.sin(angle)])
        
        return np.array(points)
    
    # Try multiple seeds with different structures
    def make_circular_seed(radius=1.0):
        points = [[0.0, 0.0]]
        for i in range(12):
            angle = i * 2 * np.pi / 12
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
        return np.array(points)
    
    def make_grid_seed():
        # 4x4 grid minus one corner, centered
        pts = []
        for i in range(4):
            for j in range(4):
                if i == 3 and j == 3:
                    continue
                pts.append([i - 1.5, j - 1.5])
        return np.array(pts)
    
    def make_ring_seed():
        points = [[0.0, 0.0]]
        # 4 inner ring
        for i in range(4):
            angle = i * np.pi / 2 + np.pi / 4
            points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        # 8 outer ring
        for i in range(8):
            angle = i * np.pi / 4
            points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        return np.array(points)
    
    # Generate candidate seeds
    seeds = [
        make_hexagonal_seed(),
        make_circular_seed(1.0),
        make_circular_seed(0.8),
        make_grid_seed(),
        make_ring_seed(),
    ]
    
    # Add some random perturbations to hexagonal seed
    rng = np.random.default_rng(42)
    for _ in range(3):
        seed = make_hexagonal_seed() + rng.normal(0, 0.1, (13, 2))
        seeds.append(seed)
    
    best_points = None
    best_score_val = -np.inf
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        
        # Normalize seed to unit area hull
        try:
            hull = ConvexHull(seed)
            hull_area = hull.volume
            if hull_area > 0:
                seed = seed / np.sqrt(hull_area)
        except Exception:
            pass
        
        # Optimize from this seed
        try:
            result = minimize(
                negative_score,
                seed.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_points = result.x.reshape(-1, 2)
            s = score(opt_points)
            
            if s > best_score_val:
                best_score_val = s
                best_points = opt_points.copy()
        except Exception as e:
            continue
    
    # If no valid result, fall back to hexagonal seed
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = make_hexagonal_seed()
        try:
            hull = ConvexHull(best_points)
            hull_area = hull.volume
            if hull_area > 0:
                best_points = best_points / np.sqrt(hull_area)
        except Exception:
            pass
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = make_hexagonal_seed()
    
    return best_points


# EVOLVE-BLOCK-END
