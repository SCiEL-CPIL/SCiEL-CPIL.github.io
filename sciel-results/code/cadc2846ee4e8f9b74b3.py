"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])

# Area of the equilateral triangle
TRI_AREA = 0.5 * np.sqrt(3) / 2.0  # Wait, area = 0.5 * base * height = 0.5 * 1 * (sqrt(3)/2) = sqrt(3)/4


def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _in_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    if y < -eps:
        return False
    if y > np.sqrt(3) / 2.0 + eps:
        return False
    # Left edge: from (0,0) to (0.5, sqrt(3)/2), line: y = sqrt(3)*x
    if x < 0 - eps:
        return False
    if x > 1.0 + eps:
        return False
    # Check if below the top edges
    # Left edge: y <= sqrt(3) * x
    if y > np.sqrt(3) * x + eps:
        return False
    # Right edge: from (1,0) to (0.5, sqrt(3)/2), line: y = -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    if y > np.sqrt(3) * (1.0 - x) + eps:
        return False
    return True


def _project_to_triangle(p):
    """Project point p onto the equilateral triangle."""
    x, y = p
    # Clamp y
    if y < 0:
        y = 0.0
    if y > np.sqrt(3) / 2.0:
        y = np.sqrt(3) / 2.0
    
    # Clamp based on edges
    # Left edge: x >= y / sqrt(3)
    if x < y / np.sqrt(3):
        x = y / np.sqrt(3)
    # Right edge: x <= 1 - y / sqrt(3)
    if x > 1.0 - y / np.sqrt(3):
        x = 1.0 - y / np.sqrt(3)
    
    return np.array([x, y])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial points: use a structured grid-like placement
    # For 11 points, we can try to place them in a pattern
    # Let's start with a reasonable initial guess
    
    # Use a 3x3 grid plus 2 extra points, or similar
    # Actually, let's use a more systematic approach:
    # Place points on a grid and then optimize
    
    # Initial configuration: distribute points roughly evenly
    # We can use a grid in barycentric coordinates
    
    # Barycentric coordinates: (u, v, w) with u+v+w=1, u,v,w >= 0
    # Point = u*A + v*B + w*C
    
    # Let's try a 3x3 grid in barycentric (9 points) plus 2 more
    # Or use a known good configuration
    
    # Start with a simple initialization
    points = []
    # Add vertices
    points.append(A.copy())
    points.append(B.copy())
    points.append(C.copy())
    
    # Add midpoints of edges
    points.append((A + B) / 2.0)
    points.append((B + C) / 2.0)
    points.append((C + A) / 2.0)
    
    # Add centroid
    centroid = (A + B + C) / 3.0
    points.append(centroid)
    
    # Add 4 more points: quarter points
    # Points at 1/4 along edges from each vertex
    for vertex in [A, B, C]:
        for other in [A, B, C]:
            if not np.allclose(vertex, other):
                # 1/4 from vertex to other
                p = 0.75 * vertex + 0.25 * other
                # Check if not already close to existing points
                too_close = False
                for existing in points:
                    if np.linalg.norm(p - existing) < 0.05:
                        too_close = True
                        break
                if not too_close:
                    points.append(p)
                if len(points) >= N:
                    break
        if len(points) >= N:
            break
    
    # If we don't have enough, add random points
    rng = np.random.RandomState(42)
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        if u + v < 1.0:
            p = u * B + v * C  # barycentric: (1-u-v)*A + u*B + v*C
            points.append(p)
    
    points = np.array(points[:N])
    
    # Project all points to be inside the triangle
    for i in range(N):
        points[i] = _project_to_triangle(points[i])
    
    # Compute initial score
    best_points = points.copy()
    best_score = _min_triangle_area(best_points) / TRI_AREA
    
    # Use optimization to improve
    # We'll use a local search / optimization approach
    
    def objective(x):
        """Minimize the negative of the minimum triangle area."""
        pts = x.reshape(-1, 2)
        # Project to triangle
        for i in range(len(pts)):
            pts[i] = _project_to_triangle(pts[i])
        min_area = _min_triangle_area(pts)
        return -min_area
    
    # Try multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Use the initial points
            x0 = best_points.flatten()
        else:
            # Perturb the best points
            perturbation = 0.1 * rng.randn(N, 2)
            x0 = (best_points + perturbation).flatten()
        
        # Use Nelder-Mead for optimization (doesn't need gradients)
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < objective(x0):
            new_points = result.x.reshape(-1, 2)
            # Project to triangle
            for i in range(N):
                new_points[i] = _project_to_triangle(new_points[i])
            
            new_score = _min_triangle_area(new_points) / TRI_AREA
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        # Check deadline
        if time.time() > deadline - 10.0:
            break
    
    return best_points


# Quick test
if __name__ == "__main__":
    pts = construct()
    print(f"Number of points: {len(pts)}")
    print(f"Minimum triangle area: {_min_triangle_area(pts)}")
    print(f"Normalized score: {_min_triangle_area(pts) / TRI_AREA}")
    print(f"Points:")
    for p in pts:
        print(f"  ({p[0]:.4f}, {p[1]:.4f})")
