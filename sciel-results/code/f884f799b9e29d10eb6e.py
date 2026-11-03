"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import linprog

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])
TRI_AREA = np.sqrt(3)/4  # area of this triangle

def _min_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |det|
                det = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
                a = 0.5 * abs(det)
                if a < min_a:
                    min_a = a
    return min_a

def _project_to_triangle(p):
    """Project a point onto the triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    # Barycentric coordinates
    # p = alpha*A + beta*B + gamma*C, alpha+beta+gamma=1
    # Solve for alpha, beta, gamma
    # p_x = beta + 0.5*gamma
    # p_y = (sqrt(3)/2)*gamma
    gamma = p[1] / (np.sqrt(3)/2)
    beta = p[0] - 0.5*gamma
    alpha = 1.0 - beta - gamma
    
    # Clip barycentric coordinates to [0,1]
    if alpha < 0:
        alpha = 0.0
    if beta < 0:
        beta = 0.0
    if gamma < 0:
        gamma = 0.0
    
    # Normalize
    s = alpha + beta + gamma
    if s < 1e-15:
        return A.copy()
    alpha /= s
    beta /= s
    gamma /= s
    
    return alpha*A + beta*B + gamma*C

def _check_inside(points, tol=1e-6):
    """Check if all points are inside the triangle."""
    for p in points:
        # Check using barycentric coordinates
        gamma = p[1] / (np.sqrt(3)/2)
        beta = p[0] - 0.5*gamma
        alpha = 1.0 - beta - gamma
        if alpha < -tol or beta < -tol or gamma < -tol:
            return False
    return True

def _solve_lp(points, t, max_iter=100):
    """
    Try to improve the minimum area by solving a sequence of LPs.
    We fix the sign of each determinant based on the current point ordering,
    then solve: maximize t subject to det(i,j,k) >= t for all triples,
    with points constrained to the triangle.
    """
    n = len(points)
    points = points.copy()
    
    # Generate all triples
    triples = list(itertools.combinations(range(n), 3))
    
    # Fix signs based on current configuration
    signs = {}
    for (i, j, k) in triples:
        p1, p2, p3 = points[i], points[j], points[k]
        det = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
        signs[(i,j,k)] = 1.0 if det >= 0 else -1.0
    
    best_points = points.copy()
    best_t = t
    
    for iteration in range(max_iter):
        # Build LP:
        # Variables: x_0, y_0, x_1, y_1, ..., x_{n-1}, y_{n-1}, t
        # Total variables: 2*n + 1
        # Maximize t (so minimize -t)
        # Constraints:
        # 1. For each triple (i,j,k): sign * (det) >= t
        #    det = (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i)
        #    This is bilinear, not linear! 
        #
        # The determinant is quadratic in the variables, so this is NOT a linear program.
        # We need a different approach.
        
        # Alternative: use gradient-based optimization with penalty
        # Or: local search with small perturbations
        
        break  # LP approach doesn't work directly due to non-linearity
    
    return best_points, best_t

def _local_search(points, t, time_limit=100.0):
    """
    Local search: try small perturbations to improve the minimum area.
    """
    start_time = time.time()
    n = len(points)
    points = points.copy()
    best_points = points.copy()
    best_t = t
    
    # Generate all triples for efficiency
    triples = list(itertools.combinations(range(n), 3))
    
    def compute_min_area(pts):
        min_a = float('inf')
        for (i, j, k) in triples:
            p1, p2, p3 = pts[i], pts[j], pts[k]
            det = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
            a = 0.5 * abs(det)
            if a < min_a:
                min_a = a
        return min_a
    
    best_t = compute_min_area(points)
    
    # Coordinate descent / random perturbation
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    rng = np.random.RandomState(42)
    
    for step_idx, step in enumerate(step_sizes):
        if time.time() - start_time > time_limit - 5:
            break
        
        for _ in range(5000):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Pick a random point to perturb
            idx = rng.randint(n)
            orig = points[idx].copy()
            
            # Random direction
            angle = rng.uniform(0, 2*np.pi)
            dx = step * np.cos(angle)
            dy = step * np.sin(angle)
            
            new_p = points[idx] + np.array([dx, dy])
            new_p = _project_to_triangle(new_p)
            
            points[idx] = new_p
            
            new_t = compute_min_area(points)
            
            if new_t > best_t + 1e-12:
                best_t = new_t
                best_points = points.copy()
            else:
                points[idx] = orig
    
    return best_points, best_t

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    # Start with a reasonable initial configuration
    # Place points on a grid-like pattern inside the triangle
    # Use barycentric coordinates for placement
    
    # Initial configuration: vertices + edge midpoints + centroid + some interior points
    # Let's use a more systematic approach: place points at various barycentric positions
    
    # Barycentric coordinates: (alpha, beta, gamma) with alpha+beta+gamma=1, all >= 0
    # Point = alpha*A + beta*B + gamma*C
    
    # Start with a hexagonal-like pattern
    barycentric_positions = [
        (1/3, 1/3, 1/3),  # centroid
        (1/2, 1/2, 0),    # midpoint of AB
        (1/2, 0, 1/2),    # midpoint of AC
        (0, 1/2, 1/2),    # midpoint of BC
        (2/3, 1/6, 1/6),  # near A
        (1/6, 2/3, 1/6),  # near B
        (1/6, 1/6, 2/3),  # near C
        (1/2, 1/4, 1/4),  # interior
        (1/4, 1/2, 1/4),  # interior
        (1/4, 1/4, 1/2),  # interior
        (2/5, 2/5, 1/5),  # interior
    ]
    
    points = []
    for (a, b, g) in barycentric_positions[:N]:
        p = a*A + b*B + g*C
        points.append(p)
    points = np.array(points)
    
    # Ensure points are inside
    for i in range(N):
        points[i] = _project_to_triangle(points[i])
    
    best_points = points.copy()
    best_t = _min_area(points)
    
    # Local search
    while time.time() - start_time < deadline - 10:
        points, t = _local_search(points, best_t, time_limit=min(30, deadline - (time.time() - start_time)))
        if t > best_t:
            best_t = t
            best_points = points.copy()
        else:
            # If no improvement, try a different perturbation strategy
            # Random restart with small perturbation
            if time.time() - start_time > deadline - 5:
                break
            # Perturb the best points slightly
            rng = np.random.RandomState(int(time.time() * 1000) % 10000)
            for i in range(N):
                angle = rng.uniform(0, 2*np.pi)
                step = rng.uniform(0.01, 0.05)
                points[i] = points[i] + step * np.array([np.cos(angle), np.sin(angle)])
                points[i] = _project_to_triangle(points[i])
    
    # Final check and return
    best_points = np.array([_project_to_triangle(p) for p in best_points])
    
    return best_points
