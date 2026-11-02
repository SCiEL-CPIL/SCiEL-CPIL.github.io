import numpy as np
from scipy.optimize import minimize

N = 14

def convex_hull_area(points):
    """
    Calculate the area of the convex hull of the given points.
    Points are assumed to be in CCW order for the hull calculation if they form a convex polygon.
    If they don't, we should compute the actual convex hull.
    For performance in the inner loop of optimization, we assume the points are the hull vertices
    and are in CCW order. If the optimizer breaks this, the penalty function handles it.
    """
    # Shoelace formula for area of polygon defined by points in order
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 distinct points.
    """
    n = len(points)
    min_area = float('inf')
    # Iterate over all unique triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = abs(cross) * 0.5
                if area < min_area:
                    min_area = area
    return min_area

def is_convex(points):
    """
    Check if the points form a strictly convex polygon in CCW order.
    """
    n = len(points)
    sign = None
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        if abs(cross) < 1e-9:
            return False # Collinear
        if sign is None:
            sign = 1 if cross > 0 else -1
        else:
            if (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                return False
    return True

def objective(params):
    """
    Objective function for optimization.
    Minimize negative score + penalty for non-convexity.
    """
    points = params.reshape(N, 2)
    
    # Check convexity
    if not is_convex(points):
        return 1e6 # Large penalty
    
    area_hull = convex_hull_area(points)
    if area_hull < 1e-9:
        return 1e6
        
    area_min_tri = min_triangle_area(points)
    
    score = area_min_tri / area_hull
    return -score

def construct():
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    initial_points = np.column_stack((x, y))
    initial_params = initial_points.flatten()
    
    # Break symmetry with small random perturbation
    rng = np.random.default_rng(42)
    perturbations = rng.normal(0, 0.01, N * 2)
    initial_params = initial_params + perturbations
    
    # Optimization
    # Nelder-Mead is good for non-smooth or constrained problems (via penalty)
    # We will run multiple starts to be safe
    best_score = -1e6
    best_params = initial_params
    
    for i in range(5): # 5 multi-starts
        if i > 0:
            # Add larger perturbation for subsequent starts
            perturbation_scale = 0.05 * (i + 1)
            perturbations = rng.normal(0, perturbation_scale, N * 2)
            start_params = initial_points.flatten() + perturbations
        else:
            start_params = initial_params
            
        result = minimize(
            objective, 
            start_params, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        # Check result
        final_points = result.x.reshape(N, 2)
        if is_convex(final_points):
            area_hull = convex_hull_area(final_points)
            if area_hull > 1e-9:
                area_min_tri = min_triangle_area(final_points)
                score = area_min_tri / area_hull
                if score > best_score:
                    best_score = score
                    best_params = result.x
                    
    # Return the best points found
    # Normalize to unit area? The task says "normalized to unit area" in the description, 
    # but the score is min_area / hull_area. The ratio is scale-invariant.
    # The evaluator measures the raw value.
    # So we can return the points as is.
    return best_params.reshape(N, 2)
