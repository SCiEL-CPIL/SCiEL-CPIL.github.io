"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        if area < min_area:
            min_area = area
    return min_area

def _objective(x):
    """Negative of the minimum triangle area (to be minimized)."""
    points = x.reshape(-1, 2)
    return -_min_triangle_area(points)

def _clip_to_triangle(points):
    """Ensure points are inside the equilateral triangle."""
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    clipped = points.copy()
    for i in range(len(clipped)):
        p = clipped[i]
        # Barycentric coordinates: p = a*v0 + b*v1 + c*v2, a+b+c=1
        # c = 2*y/sqrt(3), b = x - 0.5*c, a = 1 - b - c
        c = 2 * p[1] / np.sqrt(3)
        b = p[0] - 0.5 * c
        a = 1 - b - c
        # Clip
        eps = 1e-10
        a = max(eps, min(1 - 2*eps, a))
        b = max(eps, min(1 - 2*eps, b))
        c = max(eps, min(1 - 2*eps, c))
        s = a + b + c
        a, b, c = a/s, b/s, c/s
        clipped[i] = a*v0 + b*v1 + c*v2
    return clipped

def construct():
    # Equilateral triangle vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    # Initial configuration: well-distributed set
    init_points = []
    
    # 3 vertices
    init_points.append(v0.copy())
    init_points.append(v1.copy())
    init_points.append(v2.copy())
    
    # 3 edge midpoints
    init_points.append((v0 + v1) / 2)
    init_points.append((v1 + v2) / 2)
    init_points.append((v2 + v0) / 2)
    
    # Center
    center = (v0 + v1 + v2) / 3
    init_points.append(center.copy())
    
    # 4 more interior points
    offset = 0.12
    init_points.append(center + np.array([offset, 0]))
    init_points.append(center + np.array([-offset, 0]))
    init_points.append(center + np.array([0, offset]))
    init_points.append(center + np.array([0, -offset]))
    
    assert len(init_points) == N
    x0 = np.array(init_points).flatten()
    
    # Optimize using Powell method
    result = minimize(_objective, x0, method='Powell', 
                      options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8})
    
    points = result.x.reshape(-1, 2)
    points = _clip_to_triangle(points)
    
    return points
# EVOLVE-BLOCK-END
