"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Equilateral triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    area_triangle = np.sqrt(3)/4
    
    # Strategy: Place points on the boundary of the triangle and some interior points
    # Optimize positions to maximize the minimum triangle area
    
    # Initial configuration: distribute points around the perimeter and some interior
    # Use a parametric representation: some points on edges, some interior
    
    # Let's use a mixed approach:
    # - 3 points at the vertices (or near them)
    # - 6 points distributed along the edges
    # - 2 points in the interior
    
    # Actually, let's try a more systematic approach:
    # Place all 11 points and optimize their positions using local search
    
    # Initial guess: regular distribution
    # 3 corners, 3 midpoints of edges, 5 interior points in a pattern
    
    # Let's use a better initial configuration based on known results
    # For 11 points in an equilateral triangle, a good starting point is:
    # - 3 vertices
    # - Points along edges at regular intervals
    # - Interior points arranged in a hexagonal-like pattern
    
    # Parametrize: use barycentric coordinates or direct (x,y) with constraints
    
    # I'll use (x, y) coordinates and handle constraints via penalty or projection
    
    def min_triangle_area(points):
        """Compute the minimum area of any triangle formed by 3 points."""
        n = len(points)
        min_area = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                    if area < min_area:
                        min_area = area
        return min_area
    
    def is_inside_triangle(p):
        """Check if point p is inside or on the boundary of the triangle ABC."""
        # Barycentric coordinate method
        x, y = p
        # Triangle vertices
        x1, y1 = A
        x2, y2 = B
        x3, y3 = C
        
        # Compute barycentric coordinates
        denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
        b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
        c = 1 - a - b
        
        return (a >= -1e-9) and (b >= -1e-9) and (c >= -1e-9)
    
    def project_to_triangle(p):
        """Project point p to the nearest point inside the triangle."""
        if is_inside_triangle(p):
            return p
        
        # Find closest point on each edge
        def closest_point_on_segment(p, a, b):
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = np.clip(t, 0, 1)
            return a + t * ab
        
        edges = [(A, B), (B, C), (C, A)]
        best = None
        best_dist = float('inf')
        for a, b in edges:
            cp = closest_point_on_segment(p, a, b)
            d = np.linalg.norm(p - cp)
            if d < best_dist:
                best_dist = d
                best = cp
        return best
    
    def objective(params):
        """Objective: negative of minimum triangle area (to minimize)."""
        points = params.reshape(-1, 2)
        # Check constraints - if any point is outside, penalize
        penalty = 0.0
        for p in points:
            if not is_inside_triangle(p):
                # Project and compute distance
                proj = project_to_triangle(p)
                penalty += 1000.0 * np.linalg.norm(p - proj)
        
        if penalty > 0:
            return penalty - min_triangle_area(points)
        
        return -min_triangle_area(points)
    
    # Initial configuration
    # 3 vertices
    # 3 edge midpoints
    # 5 interior points in a pattern
    
    # Let's create a better initial guess
    # Use a grid-like or hexagonal pattern
    
    # Vertices
    p0 = A.copy()
    p1 = B.copy()
    p2 = C.copy()
    
    # Edge points (slightly inside to avoid degeneracy)
    eps = 0.05
    p3 = A + 0.5 * (B - A) + eps * np.array([0, 1])  # midpoint of AB, slightly up
    p4 = B + 0.5 * (C - B) + eps * np.array([1, 0])  # midpoint of BC, slightly right
    p5 = C + 0.5 * (A - C) + eps * np.array([0, -1])  # midpoint of CA, slightly down
    
    # Interior points - use a hexagonal-like arrangement
    # Center of triangle
    centroid = (A + B + C) / 3
    
    # 5 interior points
    # Let's place them in a pattern
    r = 0.25
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    for i in range(5):
        pass  # Will be set in the array
    
    # Build initial points array
    points_init = np.array([
        p0, p1, p2,
        p3, p4, p5,
        centroid + r * np.array([np.cos(angles[0]), np.sin(angles[0])]),
        centroid + r * np.array([np.cos(angles[1]), np.sin(angles[1])]),
        centroid + r * np.array([np.cos(angles[2]), np.sin(angles[2])]),
        centroid + r * np.array([np.cos(angles[3]), np.sin(angles[3])]),
        centroid + r * np.array([np.cos(angles[4]), np.sin(angles[4])]),
    ])
    
    # Project all points to be inside the triangle
    for i in range(len(points_init)):
        points_init[i] = project_to_triangle(points_init[i])
    
    # Optimize using Nelder-Mead
    best_params = points_init.flatten().copy()
    best_val = -min_triangle_area(points_init)
    
    # Try local optimization
    try:
        result = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if time.time() > deadline - 30:
            # Keep the initial best
            pass
        else:
            # Check if result is better
            points_result = result.x.reshape(-1, 2)
            # Project to ensure feasibility
            for i in range(len(points_result)):
                points_result[i] = project_to_triangle(points_result[i])
            
            val = min_triangle_area(points_result)
            if val > -best_val:
                best_val = -val
                best_params = points_result.flatten().copy()
    except Exception:
        pass
    
    # Convert back to points
    best_points = best_params.reshape(-1, 2)
    
    # Final projection to ensure all points are inside
    for i in range(len(best_points)):
        best_points[i] = project_to_triangle(best_points[i])
    
    # Verify and return
    return best_points

# EVOLVE-BLOCK-END
