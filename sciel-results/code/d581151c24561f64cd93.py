"""Heilbronn triangle problem: place 11 points in a unit-area triangle to maximize the smallest triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import math

N = 11

# Vertices of the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, math.sqrt(3)/2])

# Area of this triangle: 0.5 * base * height = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4
TRIANGLE_AREA = math.sqrt(3) / 4.0

def points_to_1d(points):
    """Convert 2D points to 1D array."""
    return points.flatten()

def d1_to_points(x):
    """Convert 1D array to 2D points."""
    return x.reshape(-1, 2)

def is_inside_triangle(point):
    """Check if a point is inside the triangle using barycentric coordinates."""
    x, y = point
    # Barycentric coordinates
    # For triangle with vertices V1, V2, V3
    A1 = -math.sqrt(3)/2 * x + y
    A2 = -math.sqrt(3)/2 * (1 - x) + y
    A3 = math.sqrt(3)/2 * (x - 0.5) + y  # Actually let me compute properly
    
    # Let me use a simpler check
    # The triangle is: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    if y < -1e-10:
        return False
    if y > math.sqrt(3) * x + 1e-10:
        return False
    if y > math.sqrt(3) * (1 - x) + 1e-10:
        return False
    return True

def project_into_triangle(point):
    """Project a point into the triangle if it's outside."""
    x, y = point
    # Clip to the triangle
    if y < 0:
        y = 0
    if y > math.sqrt(3) * x:
        # This is tricky, we need to find the closest point on the line y = sqrt(3)*x
        # For now, just adjust
        if x > 0.5:
            # The constraining edge is y = sqrt(3)*(1-x) for x > 0.5
            if y > math.sqrt(3) * (1 - x):
                # Project onto the edge
                # Line: sqrt(3)*x + y = sqrt(3)
                # Normal: (sqrt(3), 1)
                # Distance and projection
                d = (math.sqrt(3)*x + y - math.sqrt(3)) / (math.sqrt(3) + 1/math.sqrt(3))
                # Actually, let me just use a simple approach
                # For x in [0.5, 1], the upper bound is y <= sqrt(3)*(1-x)
                y = max(0, math.sqrt(3) * (1 - x) * 0.999)
        else:
            # For x in [0, 0.5], the upper bound is y <= sqrt(3)*x
            y = max(0, math.sqrt(3) * x * 0.999)
    
    # Final check and clipping
    if not is_inside_triangle(np.array([x, y])):
        # Fallback: project onto the centroid direction
        centroid = (V1 + V2 + V3) / 3.0
        # Move towards centroid
        for _ in range(100):
            point_new = np.array([x, y]) * 0.9 + centroid * 0.1
            if is_inside_triangle(point_new):
                return point_new
        # If still outside, return centroid
        return centroid
    
    return np.array([x, y])

def smallest_triangle_area(points):
    """Compute the smallest area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area of triangle with vertices points[i], points[j], points[k]
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0
                if area < min_area:
                    min_area = area
    return min_area

def penalty(x):
    """Penalty for points outside the triangle."""
    points = d1_to_points(x)
    penalty_val = 0.0
    for p in points:
        if not is_inside_triangle(p):
            # Compute distance to triangle boundary
            # Simple penalty: sum of violations
            if p[1] < 0:
                penalty_val += (-p[1]) ** 2
            if p[1] > math.sqrt(3) * p[0]:
                penalty_val += (p[1] - math.sqrt(3) * p[0]) ** 2
            if p[1] > math.sqrt(3) * (1 - p[0]):
                penalty_val += (p[1] - math.sqrt(3) * (1 - p[0])) ** 2
    return penalty_val

def objective(x):
    """Objective: minimize negative smallest triangle area + penalty."""
    points = d1_to_points(x)
    # Project points into triangle
    points_projected = np.array([project_into_triangle(p) for p in points])
    min_area = smallest_triangle_area(points_projected)
    pen = penalty(x)
    return -(min_area / TRIANGLE_AREA) + 1000.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Initial feasible points: use a grid-like distribution
    # Place points in a structured way
    initial_points = []
    # Corner points
    initial_points.append(V1 * 0.95)
    initial_points.append(V2 * 0.95)
    initial_points.append(V3 * 0.95 + (V1 - V3) * 0.05)
    
    # Add points in a hexagonal-like pattern
    # Use a simple approach: place remaining points on a grid
    remaining = N - 3
    for i in range(remaining):
        # Place in a structured pattern
        u = (i + 1) / (remaining + 1)
        # Place along a line or in a pattern
        # Let me use a simple barycentric grid
        # For 8 remaining points, place them in rows
        row = i // 3
        col = i % 3
        # Barycentric coordinates
        if row == 0:
            # First row (3 points)
            t = (col + 1) / 4
            point = V1 + t * (V2 - V1)
        elif row == 1:
            # Second row (3 points)
            t = (col + 1) / 4
            point = (V1 + t * (V2 - V1)) * 0.5 + (V3 * 0.5)
        else:
            # Third row (2 points)
            t = (col + 1) / 3
            point = (V1 + t * (V2 - V1)) * 0.25 + (V3 * 0.75)
        
        # Scale to be inside
        point = point * 0.9
        initial_points.append(point)
    
    initial_points = np.array(initial_points)
    # Ensure all are inside
    initial_points = np.array([project_into_triangle(p) for p in initial_points])
    
    # Evaluate initial solution
    current_best = initial_points.copy()
    current_best_score = smallest_triangle_area(current_best) / TRIANGLE_AREA
    
    # Define bounds for differential evolution
    # The triangle fits in [0, 1] x [0, sqrt(3)/2]
    bounds = [(0.0, 1.0)] * N + [(0.0, math.sqrt(3)/2)] * N
    # Actually, the bounds should be per-coordinate
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x coordinate
        bounds.append((0.0, math.sqrt(3)/2))  # y coordinate
    
    # Differential evolution
    best_x = None
    best_val = float('inf')
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            x0=points_to_1d(initial_points),
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            init='sobol',
            polish=True
        )
        
        if time.time() < deadline:
            best_x = result.x
            best_val = result.fun
            
            # Project the result
            best_points = np.array([project_into_triangle(p) for p in d1_to_points(best_x)])
            best_area = smallest_triangle_area(best_points) / TRIANGLE_AREA
            
            if best_area > current_best_score:
                current_best = best_points
                current_best_score = best_area
    except Exception as e:
        pass
    
    # Local refinement using Nelder-Mead on the best solution
    if time.time() < deadline - 10:
        def local_objective(x):
            points = d1_to_points(x)
            points_projected = np.array([project_into_triangle(p) for p in points])
            min_area = smallest_triangle_area(points_projected)
            return -min_area / TRIANGLE_AREA
        
        try:
            result_lm = minimize(
                local_objective,
                points_to_1d(current_best),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if time.time() < deadline - 5:
                refined_points = np.array([project_into_triangle(p) for p in d1_to_points(result_lm.x)])
                refined_area = smallest_triangle_area(refined_points) / TRIANGLE_AREA
                
                if refined_area > current_best_score:
                    current_best = refined_points
                    current_best_score = refined_area
        except Exception as e:
            pass
    
    # Final check
    if time.time() > deadline:
        pass
    
    return current_best
