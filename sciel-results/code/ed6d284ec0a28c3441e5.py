"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _min_triangle_area_normalized(points):
    """
    Compute the minimum area of any triangle formed by 3 points,
    normalized by the convex hull area.
    
    points: (N, 2) array
    Returns: float, the normalized min triangle area (higher is better)
    """
    n = len(points)
    # Compute convex hull area
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if hull fails, use a large penalty
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Find minimum triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _objective(params):
    """
    Objective function for optimization: minimize negative of min normalized triangle area.
    
    params: (N,) array of angles (radians) on a circle
    Returns: float, negative of the normalized min triangle area
    """
    # Convert angles to points on unit circle
    angles = params
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    score = _min_triangle_area_normalized(points)
    return -score


def construct():
    """
    Construct N points in a planar convex region to maximize the minimum
    triangle area normalized by convex hull area.
    
    Returns: (N, 2) array of points
    """
    # Start with equally spaced angles (regular polygon)
    initial_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Use Nelder-Mead to optimize the angles
    result = minimize(
        _objective,
        initial_angles,
        method='Nelder-Mead',
        options={
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    optimal_angles = result.x
    points = np.column_stack([np.cos(optimal_angles), np.sin(optimal_angles)])
    
    return points
# EVOLVE-BLOCK-END
