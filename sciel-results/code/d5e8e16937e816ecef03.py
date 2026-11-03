"""Heilbronn triangles: place 11 points to maximize the smallest triangle area."""
N = 11
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_flat(x):
    points = x.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            penalty += 1.0
    return -min_triangle_area(points) + penalty

def project_to_triangle(p):
    """Project point p onto the triangle (0,0),(1,0),(0.5,sqrt(3)/2) if outside."""
    if point_in_triangle(p):
        return p
    
    # Use barycentric coordinates to find the closest point
    # Barycentric coordinates: p = a*V0 + b*V1 + c*V2 where a+b+c=1
    # For triangle with V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2):
    # p = b*V1 + c*V2
    # px = b + 0.5*c
    # py = sqrt(3)/2 * c
    # c = py / (sqrt(3)/2) = 2*py/sqrt(3)
    # b = px - 0.5*c
    
    px, py = p
    c = 2 * py / np.sqrt(3)
    b = px - 0.5 * c
    a = 1 - b - c
    
    # Clamp to [0,1]
    a = max(0, a)
    b = max(0, b)
    c = max(0, c)
    
    # Renormalize
    s = a + b + c
    if s < 1e-15:
        return V0.copy()
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    def initial_points(seed_offset=0):
        """Generate initial points using a structured pattern."""
        rng_local = np.random.RandomState(42 + seed_offset)
        pts = []
        
        # Place points near vertices
        pts.append(V0 + 0.05 * np.array([0.5, np.sqrt(3)/6]))
        pts.append(V1 + 0.05 * np.array([-0.5, np.sqrt(3)/6]))
        pts.append(V2 + 0.05 * np.array([0, -np.sqrt(3)/3]))
        
        # Add more points in a pattern
        for i in range(8):
            u = rng_local.random() * 0.85 + 0.05
            v = rng_local.random() * (1 - u) * 0.85 + 0.05
            p = u * V1 + v * V2
            pts.append(p)
        return np.array(pts)
    
    best_points = None
    best_score = -float('inf')
    
    # Try multiple initial configurations with L-BFGS-B
    for restart in range(50):
        if time.time() > deadline - 15.0:
            break
        
        points = initial_points(restart)
        x0 = points.flatten()
        
        # Use L-BFGS-B with bounds
        # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
        # But we need to ensure points are inside the triangle
        bounds = [(0, 1), (0, np.sqrt(3)/2)] * 11
        
        try:
            result = minimize(
                negative_min_area_flat,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
        except Exception:
            continue
        
        if time.time() > deadline - 10.0:
            break
        
        optimized_points = result.x.reshape(-1, 2)
        
        # Project any points outside the triangle back inside
        for i in range(len(optimized_points)):
            if not point_in_triangle(optimized_points[i]):
                optimized_points[i] = project_to_triangle(optimized_points[i])
        
        # Check if all points are inside triangle
        valid = all(point_in_triangle(p) for p in optimized_points)
        
        if valid:
            score = min_triangle_area(optimized_points)
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
    
    # Also try Nelder-Mead on the best result found so far
    if best_points is not None and time.time() < deadline - 10.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                negative_min_area_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            optimized_points = result.x.reshape(-1, 2)
            for i in range(len(optimized_points)):
                if not point_in_triangle(optimized_points[i]):
                    optimized_points[i] = project_to_triangle(optimized_points[i])
            valid = all(point_in_triangle(p) for p in optimized_points)
            if valid:
                score = min_triangle_area(optimized_points)
                if score > best_score:
                    best_score = score
                    best_points = optimized_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return initial points
        best_points = initial_points(0)
    
    return best_points
