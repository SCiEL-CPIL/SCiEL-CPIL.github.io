"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle with side length 1
# Area = sqrt(3)/4
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
TRIANGLE_AREA = np.sqrt(3) / 4

def _barycentric_coords(p):
    """Compute barycentric coordinates of point p with respect to triangle ABC."""
    v0 = B - A
    v1 = C - A
    v2 = p - A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def _project_to_triangle(p):
    """Project point p onto the triangle ABC if it's outside."""
    u, v, w = _barycentric_coords(p)
    if u >= -1e-10 and v >= -1e-10 and w >= -1e-10:
        return p
    # Project to nearest edge
    # Edge AB: v=0, so p = A + t*(B-A)
    # Edge BC: u=0
    # Edge CA: w=0
    best_p = p
    best_dist = np.inf
    
    # Edge AB
    t = np.clip(np.dot(p - A, B - A) / np.dot(B - A, B - A), 0, 1)
    p_ab = A + t * (B - A)
    d = np.linalg.norm(p - p_ab)
    if d < best_dist:
        best_dist = d
        best_p = p_ab
    
    # Edge BC
    t = np.clip(np.dot(p - B, C - B) / np.dot(C - B, C - B), 0, 1)
    p_bc = B + t * (C - B)
    d = np.linalg.norm(p - p_bc)
    if d < best_dist:
        best_dist = d
        best_p = p_bc
    
    # Edge CA
    t = np.clip(np.dot(p - C, A - C) / np.dot(A - C, A - C), 0, 1)
    p_ca = C + t * (A - C)
    d = np.linalg.norm(p - p_ca)
    if d < best_dist:
        best_dist = d
        best_p = p_ca
    
    return best_p

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    min_area = np.inf
    for i, j, k in combinations(range(len(points)), 3):
        area = _triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
    return min_area

def _objective(x):
    """Objective function: negative of minimum triangle area."""
    points = np.array(x).reshape(-1, 2)
    # Project points to triangle if needed
    points = np.array([_project_to_triangle(p) for p in points])
    return -_min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_value = -np.inf
    
    # Initial good starting points: use a grid-like distribution
    # For 11 points, try several initial configurations
    
    # Configuration 1: Random with seed
    rng = np.random.RandomState(42)
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if attempt == 0:
            # Uniform random
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v <= 1.0:
                        points[i] = [u + 0.5 * v, np.sqrt(3) / 2 * v]
                        break
        elif attempt == 1:
            # Grid-based: 3x3 grid plus 2 extra
            points = []
            for i in range(3):
                for j in range(3):
                    if i + j <= 2:
                        # Map grid to triangle
                        u = (i + 0.5) / 3
                        v = (j + 0.5) / 3
                        if u + v <= 1.0:
                            points.append([u + 0.5 * v, np.sqrt(3) / 2 * v])
            # Add 2 more points
            while len(points) < N:
                u, v = rng.random(), rng.random()
                if u + v <= 1.0:
                    points.append([u + 0.5 * v, np.sqrt(3) / 2 * v])
            points = np.array(points[:N])
        elif attempt == 2:
            # Symmetric initial: vertices + midpoints + center + 2 more
            points = [
                [0.05, 0.05],
                [0.95, 0.05],
                [0.5, np.sqrt(3)/2 - 0.05],
                [0.5, 0.05],
                [0.25, np.sqrt(3)/4],
                [0.75, np.sqrt(3)/4],
                [0.5, np.sqrt(3)/2 * 0.3],
                [0.5, np.sqrt(3)/2 * 0.7],
                [0.3, np.sqrt(3)/6],
                [0.7, np.sqrt(3)/6],
                [0.5, np.sqrt(3)/3],
            ]
            points = np.array(points)
        else:
            # Perturbed versions of previous best
            if best_points is not None:
                points = best_points + rng.normal(0, 0.05, (N, 2))
                points = np.array([_project_to_triangle(p) for p in points])
            else:
                continue
        
        # Optimize
        x0 = points.flatten()
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        except Exception:
            continue
        
        opt_points = np.array(result.x).reshape(-1, 2)
        opt_points = np.array([_project_to_triangle(p) for p in opt_points])
        opt_value = _min_triangle_area(opt_points)
        
        if opt_value > best_value:
            best_value = opt_value
            best_points = opt_points
        
        # Local refinement with L-BFGS-B
        if time.time() > deadline:
            break
        
        try:
            result2 = minimize(_objective, x0, method='L-BFGS-B',
                              options={'maxiter': 2000})
        except Exception:
            pass
        else:
            opt_points2 = np.array(result2.x).reshape(-1, 2)
            opt_points2 = np.array([_project_to_triangle(p) for p in opt_points2])
            opt_value2 = _min_triangle_area(opt_points2)
            
            if opt_value2 > best_value:
                best_value = opt_value2
                best_points = opt_points2
    
    if best_points is None:
        # Fallback: random points
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, np.sqrt(3) / 2 * v])
        best_points = np.array(points)
    
    return best_points
