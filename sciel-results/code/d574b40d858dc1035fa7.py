"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 11

# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.optimize import minimize
import time

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by N points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _inside_triangle(p, margin=0.001):
    """Check if point p is inside the equilateral triangle with margin."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric coordinates
    h = np.sqrt(3) / 2
    # b1 = area of triangle (p, v2, v3) / total area
    # b2 = area of triangle (v1, p, v3) / total area  
    # b3 = area of triangle (v1, v2, p) / total area
    
    # Using the formula: for point (x,y) in triangle with vertices (0,0), (1,0), (0.5, h)
    # b3 (weight for v1=(0,0)) = 1 - x - y/h  ... wait let me recompute
    # Actually: barycentric coordinates
    # v1 = (0,0), v2 = (1,0), v3 = (0.5, h)
    # For point p = (x, y):
    # b1 (for v1) = area(p, v2, v3) / area(v1, v2, v3)
    # b2 (for v2) = area(v1, p, v3) / area(v1, v2, v3)
    # b3 (for v3) = area(v1, v2, p) / area(v1, v2, v3)
    
    total_area = h / 2  # area of unit equilateral triangle = sqrt(3)/4, but h/2 = sqrt(3)/4
    
    # area(p, v2, v3):
    # p=(x,y), v2=(1,0), v3=(0.5,h)
    a1 = 0.5 * abs((1-x)*(h-y) - (0.5-x)*(0-y))
    a1 = 0.5 * abs((1-x)*h - (0.5-x)*y)
    
    # area(v1, p, v3):
    # v1=(0,0), p=(x,y), v3=(0.5,h)
    a2 = 0.5 * abs(x*h - 0.5*y)
    
    # area(v1, v2, p):
    # v1=(0,0), v2=(1,0), p=(x,y)
    a3 = 0.5 * abs(1*y - 0*x)
    a3 = 0.5 * abs(y)
    
    b1 = a1 / total_area
    b2 = a2 / total_area
    b3 = a3 / total_area
    
    return b1 >= margin and b2 >= margin and b3 >= margin

def _project_inside(p):
    """Project point inside the triangle if it's outside."""
    x, y = p
    h = np.sqrt(3) / 2
    # Simple projection: clip to triangle
    # For now, just return the point; we'll use constraints in optimization
    return np.array([x, y])

def _objective_neg_min_area(params):
    """Objective: negative of minimum triangle area (we minimize this)."""
    points = params.reshape(-1, 2)
    return -_min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    h = np.sqrt(3) / 2
    vertices = np.array([[0, 0], [1, 0], [0.5, h]])
    
    # Generate initial points using a structured approach
    # Start with a good initial configuration: vertices + centroid + edge midpoints + interior points
    
    def initial_points():
        pts = []
        # Vertices
        pts.append([0.05, 0.05])
        pts.append([0.95, 0.05])
        pts.append([0.5, h - 0.05])
        # Edge midpoints (slightly inside)
        pts.append([0.5, 0.02])
        pts.append([0.75, h/4 + 0.02])
        pts.append([0.25, h/4 + 0.02])
        # Centroid
        pts.append([0.5, h/3])
        # Additional interior points
        pts.append([0.35, h/3 + 0.05])
        pts.append([0.65, h/3 + 0.05])
        pts.append([0.5, h/3 - 0.08])
        # One more
        pts.append([0.4, h/5 + 0.1])
        return np.array(pts[:N])
    
    best_points = initial_points()
    best_value = _min_triangle_area(best_points)
    
    # Local optimization using Nelder-Mead
    # We'll do multiple restarts with perturbed initial conditions
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = best_points.flatten().copy()
        else:
            # Perturb the best solution
            x0 = best_points.flatten().copy()
            noise = np.random.randn(N * 2) * 0.05
            x0 += noise
            # Project back inside triangle (simple clipping)
            for i in range(N):
                px, py = x0[2*i], x0[2*i+1]
                # Clip to be inside triangle
                # Simple approach: if outside, move toward centroid
                cx, cy = 0.5, h/3
                if not _inside_triangle([px, py], margin=0.005):
                    t = 0.5
                    x0[2*i] = px + t * (cx - px)
                    x0[2*i+1] = py + t * (cy - py)
        
        # Use Nelder-Mead to maximize min area
        result = minimize(
            _objective_neg_min_area,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.success or result.fun < -best_value:
            new_points = result.x.reshape(-1, 2)
            # Verify all points are inside triangle
            valid = all(_inside_triangle(p, margin=0.001) for p in new_points)
            if valid:
                new_val = _min_triangle_area(new_points)
                if new_val > best_value:
                    best_value = new_val
                    best_points = new_points.copy()
        
        # Also try a second pass with finer tolerance from the current best
        if time.time() > deadline:
            break
        
        # Fine-tune
        result2 = minimize(
            _objective_neg_min_area,
            best_points.flatten().copy(),
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        new_points = result2.x.reshape(-1, 2)
        valid = all(_inside_triangle(p, margin=0.001) for p in new_points)
        if valid:
            new_val = _min_triangle_area(new_points)
            if new_val > best_value:
                best_value = new_val
                best_points = new_points.copy()
    
    # Final check
    if time.time() > deadline - 5:
        pass  # Return whatever we have
    
    return best_points
# EVOLVE-BLOCK-END
