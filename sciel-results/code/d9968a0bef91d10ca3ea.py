"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _triangle_area(p, q, r):
    """Compute area of triangle formed by points p, q, r."""
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))

def _all_triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def _min_area(points):
    """Return the minimum triangle area among all triples."""
    return np.min(_all_triangle_areas(points))

def _inside_triangle(pt, tol=1e-9):
    """Check if point is inside or on the boundary of the equilateral triangle."""
    x, y = pt
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Conditions:
    # 1. y >= 0
    # 2. y <= sqrt(3)/2
    # 3. y <= sqrt(3) * x  (left side)
    # 4. y <= sqrt(3) * (1 - x)  (right side)
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2
    if y < -tol:
        return False
    if y > h + tol:
        return False
    if y > sqrt3 * x + tol:
        return False
    if y > sqrt3 * (1 - x) + tol:
        return False
    return True

def _clamp_to_triangle(pt):
    """Project a point onto the equilateral triangle if outside."""
    x, y = pt
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2
    
    # Clamp y
    if y < 0:
        y = 0
    if y > h:
        y = h
    
    # For a given y, x must satisfy:
    # y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    # y <= sqrt(3) * (1 - x)  =>  x <= 1 - y / sqrt(3)
    
    x_min = y / sqrt3
    x_max = 1 - y / sqrt3
    
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    
    return np.array([x, y])

def _objective(points_flat):
    """Objective function to minimize: negative of min triangle area."""
    points = points_flat.reshape(-1, 2)
    # Clamp points to triangle
    points = np.array([_clamp_to_triangle(p) for p in points])
    min_a = _min_area(points)
    return -min_a

def _constraint_penalty(points_flat, mu=1.0):
    """Penalty for points outside the triangle."""
    points = points_flat.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        if not _inside_triangle(p):
            # Compute distance to triangle
            clamped = _clamp_to_triangle(p)
            penalty += mu * np.sum((p - clamped) ** 2)
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2
    
    # Initial guess: 11 points spread in the triangle
    # Use a grid-like initial placement
    def initial_guess():
        # Place points in a hexagonal-like pattern
        pts = []
        # Center
        pts.append([0.5, h / 2])
        # Ring 1
        r1 = 0.2
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            x = 0.5 + r1 * np.cos(angle)
            y = h / 2 + r1 * np.sin(angle)
            pts.append([x, y])
        # Ring 2
        r2 = 0.35
        for angle in np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, 5, endpoint=False):
            x = 0.5 + r2 * np.cos(angle)
            y = h / 2 + r2 * np.sin(angle)
            pts.append([x, y])
        
        # Clamp all to triangle
        pts = np.array([_clamp_to_triangle(p) for p in pts])
        return pts
    
    best_points = initial_guess()
    best_min_area = _min_area(best_points)
    
    # Local optimization using Nelder-Mead
    def _optimize_from(points_flat, maxiter=500):
        """Run Nelder-Mead optimization from given starting point."""
        n_params = 2 * N
        
        def obj(x):
            return _objective(x)
        
        result = minimize(
            obj,
            points_flat,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        return result.x, result.fun
    
    # Multiple restarts
    rng = np.random.RandomState(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate random starting points
        if restart == 0:
            start_pts = best_points
        else:
            # Random points inside triangle
            start_pts = []
            for _ in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = u + 0.5 * v
                y = h * v
                start_pts.append([x, y])
            start_pts = np.array([_clamp_to_triangle(p) for p in start_pts])
        
        start_flat = start_pts.flatten()
        
        try:
            opt_flat, fun_val = _optimize_from(start_flat, maxiter=300)
        except Exception:
            continue
        
        opt_pts = np.array([_clamp_to_triangle(p) for p in opt_flat.reshape(-1, 2)])
        cur_min_area = _min_area(opt_pts)
        
        if cur_min_area > best_min_area:
            best_min_area = cur_min_area
            best_points = opt_pts
    
    return best_points
# EVOLVE-BLOCK-END
