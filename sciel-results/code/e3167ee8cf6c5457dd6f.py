"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    # Vectorized computation for speed
    # points: (n, 2)
    # We need min over i<j<k of 0.5 * |det(points[j]-points[i], points[k]-points[i])|
    # For n=14, C(14,3) = 364 triangles. We can compute all areas.
    
    # Efficient vectorized approach:
    # For each pair (i,j), compute cross product with all other points k
    # But simpler: just iterate since n is small
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            v1 = points[j] - points[i]
            for k in range(j+1, n):
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except:
        # Fallback: use shoelace on the points assuming they are in convex position
        # This is a rough fallback
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
        return abs(area) / 2.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _negative_score(points):
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_points = None
    best_score = -1.0
    
    # Initialize with regular polygon (a good starting point for this problem)
    def make_regular_polygon(n, radius=1.0, angle_offset=0.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + angle_offset
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.column_stack([x, y])
    
    # Start with a regular 14-gon
    initial_points = make_regular_polygon(N)
    best_points = initial_points.copy()
    best_score = _score(best_points)
    
    # Try multiple random perturbations and local optimizations
    rng = np.random.RandomState(42)
    
    n_attempts = 50
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Create a perturbed version of the best solution or a random one
        if attempt < 10:
            # Small perturbations of the current best
            perturbation_scale = 0.1 * (1 - attempt/10.0)
            points = best_points + rng.randn(N, 2) * perturbation_scale
        elif attempt < 30:
            # Medium perturbations
            perturbation_scale = 0.3 * rng.random()
            points = best_points + rng.randn(N, 2) * perturbation_scale
        else:
            # Larger random restarts
            if attempt % 5 == 0:
                # Regular polygon with random rotation
                points = make_regular_polygon(N, radius=1.0 + 0.5*rng.random(), 
                                             angle_offset=rng.random()*2*np.pi)
            else:
                # Random points in unit circle
                angles = rng.uniform(0, 2*np.pi, N)
                radii = rng.uniform(0.3, 1.0, N)
                points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Local optimization using L-BFGS-B
        try:
            result = minimize(_negative_score, points, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < _negative_score(best_points):
                candidate = result.x
                score = _score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        except Exception as e:
            pass
    
    # Final polish: try a few more iterations on the best solution
    if time.time() < deadline - 5:
        try:
            result = minimize(_negative_score, best_points, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < _negative_score(best_points):
                candidate = result.x
                score = _score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        except:
            pass
    
    # Normalize: the score is already normalized by convex hull area
    return best_points
# EVOLVE-BLOCK-END
