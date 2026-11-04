"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import math

def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3 (2D points)."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                     (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        a = _triangle_area(points[i], points[j], points[k])
        if a < min_area:
            min_area = a
    return min_area

def _inside_triangle(p, margin=1e-9):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    h = math.sqrt(3) / 2
    if y < -margin:
        return False
    if x < -margin or x > 1 + margin:
        return False
    if y > math.sqrt(3) * x + margin:
        return False
    if y > math.sqrt(3) * (1 - x) + margin:
        return False
    return True

def _project_to_triangle(p, margin=1e-8):
    """Project point p onto the triangle (0,0),(1,0),(0.5,sqrt(3)/2) with margin."""
    x, y = p
    h = math.sqrt(3) / 2
    if y < margin:
        y = margin
    min_x = y / math.sqrt(3)
    if x < min_x:
        x = min_x
    max_x = 1 - y / math.sqrt(3)
    if x > max_x:
        x = max_x
    if x < margin:
        x = margin
    if x > 1 - margin:
        x = 1 - margin
    return [x, y]

def _neg_min_area_flat(flat_points):
    """Negative of minimum triangle area for optimization (minimize)."""
    points = flat_points.reshape(N, 2)
    points = np.array([_project_to_triangle(p) for p in points])
    min_a = _min_triangle_area(points)
    return -min_a

def _initial_guess_symmetric():
    """Generate a symmetric initial guess."""
    h = math.sqrt(3) / 2
    cx, cy = 0.5, h / 3
    # 3 vertices
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h]
    ]
    # 1 center
    pts.append([cx, cy])
    # 3 points on medians at 1/3 from center towards vertices?
    # Or just a ring?
    # Let's place 7 more points in a symmetric pattern.
    # 3 points at midpoints of edges?
    pts.append([0.25, h/2]) # Mid left? No, (0.25, sqrt(3)/4)
    pts.append([0.75, h/2])
    pts.append([0.5, h/2]) # Center of top edge? No, that's on boundary.
    
    # Let's try a grid-like symmetric distribution
    # Row 1 (bottom): 3 points
    # Row 2 (middle): 3 points
    # Row 3 (top): 1 point (center)
    # Plus 3 vertices?
    
    # Better: 3 vertices, 3 edge midpoints, 3 inner ring, 1 center.
    # Vertices:
    p0 = [0.0, 0.0]
    p1 = [1.0, 0.0]
    p2 = [0.5, h]
    
    # Edge midpoints:
    m0 = [0.5, 0.0]
    m1 = [0.75, h/2]
    m2 = [0.25, h/2]
    
    # Inner points:
    # Center
    c = [0.5, h/3]
    
    # 3 more points to make 11? We have 3+3+1 = 7. Need 4 more.
    # Let's place them at 1/3 and 2/3 along medians?
    # Median from (0,0) to (0.5, h): points at (1/6, h/6) and (1/3, h/3)
    # Median from (1,0) to (0.5, h): points at (5/6, h/6) and (2/3, h/3)
    # Median from (0.5,h) to (0.5,0): points at (0.5, 2h/3) and (0.5, h/3) (center already there)
    
    pts = [p0, p1, p2, m0, m1, m2, c]
    pts.append([1/6, h/6])
    pts.append([5/6, h/6])
    pts.append([0.5, 2*h/3])
    pts.append([0.25, h/4]) # Just a guess
    
    return np.array(pts)

def construct():
    """Construct 11 points in the equilateral triangle maximizing min triangle area."""
    h = math.sqrt(3) / 2
    
    # 1. Generate symmetric initial guess
    init_pts = _initial_guess_symmetric()
    
    # 2. Differential Evolution for global optimization
    # Bounds: x in [0,1], y in [0, h]
    bounds = [(0, 1), (0, h)] * N
    # We need to project inside, so DE will explore the box, and we project.
    
    # DE is slow, use small population and few generations for a quick global search
    # Then polish.
    try:
        res_de = differential_evolution(
            _neg_min_area_flat, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4,
            init='sobol'
        )
        best_flat = res_de.x
        best_val = res_de.fun
    except Exception:
        best_flat = init_pts.flatten()
        best_val = _neg_min_area_flat(best_flat)
        
    # 3. Polish with Nelder-Mead
    res_nm = minimize(_neg_min_area_flat, best_flat, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
    if res_nm.fun < best_val:
        best_val = res_nm.fun
        best_flat = res_nm.x
        
    # 4. Try a few random starts with Nelder-Mead just in case DE failed
    rng = np.random.RandomState(42)
    for trial in range(5):
        pts = []
        for _ in range(N):
            u, v = rng.random(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            p = [u + 0.5 * v, h * v]
            pts.append(_project_to_triangle(p))
        flat0 = np.array(pts).flatten()
        res = minimize(_neg_min_area_flat, flat0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_flat = res.x

    points = best_flat.reshape(N, 2)
    points = np.array([_project_to_triangle(p) for p in points])
    
    return points
# EVOLVE-BLOCK-END
