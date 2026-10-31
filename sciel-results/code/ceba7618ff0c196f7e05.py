"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    h = np.sqrt(3) / 2
    v0 = np.array([1, 0])
    v1 = np.array([0.5, h])
    v2 = np.array([p[0], p[1]])
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return False
    
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    
    return u >= -tol and v >= -tol and (u + v) <= 1 + tol

def _is_valid(points):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p) for p in points)

def _objective(points_flat):
    """Objective function to minimize the negative of the minimum triangle area."""
    points = points_flat.reshape(-1, 2)
    if not _is_valid(points):
        # Soft penalty: sum of squared distances to boundary
        penalty = 0.0
        h = np.sqrt(3) / 2
        for p in points:
            d1 = p[1]
            d2 = 1.0 - p[0]
            d3 = (np.sqrt(3) * p[0] - p[1]) / 2.0
            if d1 < 0: penalty += d1 * d1
            if d2 < 0: penalty += d2 * d2
            if d3 < 0: penalty += d3 * d3
        return 1e5 + penalty
        
    min_area = _min_triangle_area(points)
    return -min_area

def _hex_initial_guess():
    """Generate initial points from a hexagonal lattice pattern."""
    h = np.sqrt(3) / 2
    pts = []
    # 4 rows: y=0, h/3, 2h/3, h
    for j in range(4):
        y = j * h / 3.0
        x_min = 0.5 * j / 3.0
        x_max = 1.0 - 0.5 * j / 3.0
        if x_min > x_max:
            continue
        n_pts = 4 - j
        if n_pts <= 0: continue
        for i in range(n_pts):
            if n_pts == 1:
                x = 0.5
            else:
                x = x_min + (x_max - x_min) * i / (n_pts - 1)
            pts.append([x, y])
            
    # We have 10 points. Add one more.
    # Place it in the center of the largest gap.
    # For simplicity, add a point near the center (0.5, h/2) with slight offset.
    import random
    rng = random.Random(42)
    for _ in range(100):
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            p = [u + 0.5*v, h*v]
            min_dist = min(np.linalg.norm(np.array(p) - np.array(q)) for q in pts)
            if min_dist > 0.05:
                pts.append(p)
                break
                
    return np.array(pts).flatten()

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the smallest triangle area."""
    h = np.sqrt(3) / 2
    
    # Get initial guess
    x0 = _hex_initial_guess()
    
    # Define bounds for each coordinate
    bounds = [(0, 1), (0, h)] * N
    
    # 1. Global Search with Differential Evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            polish=False,
            init='sobol',
            popsize=15
        )
        best_points = result.x.reshape(-1, 2)
    except Exception:
        best_points = x0.reshape(-1, 2)
    
    # 2. Local Refinement with Nelder-Mead
    if _is_valid(best_points):
        res = minimize(
            _objective,
            best_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if _is_valid(res.x.reshape(-1, 2)):
            best_points = res.x.reshape(-1, 2)
            
    # 3. Final Validation
    if not _is_valid(best_points):
        import random
        rng = random.Random(7)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, h * v])
        best_points = np.array(pts)
    
    return best_points
# EVOLVE-BLOCK-END
