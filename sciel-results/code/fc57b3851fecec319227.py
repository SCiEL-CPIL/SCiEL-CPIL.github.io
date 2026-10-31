"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _evaluate_config(params):
    """
    Generate points from parameters and return the negative ratio.
    params: [r1, r2, theta1, theta2]
    Configuration: 1 center, 6 inner, 6 outer.
    """
    r1, r2, theta1, theta2 = params
    
    # Center
    pts = [(0.0, 0.0)]
    
    # Inner ring (6 points)
    angles1 = theta1 + np.arange(6) * (2 * np.pi / 6)
    for a in angles1:
        pts.append((r1 * np.cos(a), r1 * np.sin(a)))
        
    # Outer ring (6 points)
    angles2 = theta2 + np.arange(6) * (2 * np.pi / 6)
    for a in angles2:
        pts.append((r2 * np.cos(a), r2 * np.sin(a)))
        
    points = np.array(pts)
    
    # Check for degeneracy
    if r1 < 1e-6 or r2 < 1e-6:
        return 1.0 # Penalty
        
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1.0
        
    min_area = _min_triangle_area(points)
    ratio = min_area / hull_area
    return -ratio

def construct():
    """Construct 13 points in a planar convex region to maximize the minimum triangle area ratio."""
    
    # Define the parameter space
    # r1, r2 are radii. We can assume 0 < r1 < r2.
    # theta1, theta2 are angles. Due to symmetry, we can restrict ranges.
    # Let's allow r1 in [0.1, 1.0], r2 in [r1+0.1, 2.0] (scale doesn't matter for ratio)
    # Actually, since ratio is scale invariant, we can fix r2=1.0 and optimize r1, theta1, theta2.
    # This reduces dimensionality further.
    
    def objective_fixed_r2(params):
        r1, theta1, theta2 = params
        # Ensure r1 is positive
        if r1 < 0.01:
            return 1.0
        return _evaluate_config([r1, 1.0, theta1, theta2])
        
    # Initial guesses
    # Regular hexagon on unit circle: r1=1, theta=0. But we have two rings.
    # Try r1 = 0.5, theta1 = 0, theta2 = 0
    x0 = [0.5, 0.0, 0.0]
    
    # Bounds: r1 in (0, 1), angles in (0, pi/3) due to 6-fold symmetry
    bounds = [(0.01, 0.99), (0, np.pi/6), (0, np.pi/6)]
    
    # Optimize
    result = minimize(
        objective_fixed_r2,
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    # Extract best params
    r1, theta1, theta2 = result.x
    
    # Generate final points
    pts = [(0.0, 0.0)]
    angles1 = theta1 + np.arange(6) * (2 * np.pi / 6)
    for a in angles1:
        pts.append((r1 * np.cos(a), r1 * np.sin(a)))
        
    angles2 = theta2 + np.arange(6) * (2 * np.pi / 6)
    for a in angles2:
        pts.append((1.0 * np.cos(a), 1.0 * np.sin(a)))
        
    points = np.array(pts)
    
    # Normalize to unit area convex hull
    hull_area = _convex_hull_area(points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        points = points * scale
        
    return points
# EVOLVE-BLOCK-END
