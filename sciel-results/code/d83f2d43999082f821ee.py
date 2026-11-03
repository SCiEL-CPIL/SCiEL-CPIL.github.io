"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

def point_in_triangle(p, a=V1, b=V2, c=V3):
    """Check if point p is inside or on the triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Minimum area of any triangle formed by three points."""
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

def objective(params):
    """Objective function for minimization: -min_area + penalty."""
    points = params.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            penalty += 10.0
    if penalty > 0:
        return penalty - min_triangle_area(points)
    return -min_triangle_area(points)

def construct():
    # Define a few structured initial configurations
    inits = []
    
    def gen_grid(rows):
        pts = []
        total_rows = len(rows)
        # Distribute y evenly, keeping margin from boundaries
        for i, count in enumerate(rows):
            y = (i + 1) * (0.7 / total_rows) + 0.15
            # x distribution
            if count == 1:
                pts.append(np.array([0.5, y]))
            else:
                for j in range(count):
                    x = 0.25 + j * (0.5 / (count - 1))
                    pts.append(np.array([x, y]))
        return np.array(pts)

    # Try different row distributions
    inits.append(gen_grid([4, 4, 3]))
    inits.append(gen_grid([3, 4, 4]))
    inits.append(gen_grid([4, 3, 4]))
    inits.append(gen_grid([2, 4, 5]))
    inits.append(gen_grid([5, 4, 2]))
    
    # Add a random uniform start
    np.random.seed(42)
    r1 = np.sqrt(np.random.rand(N))
    r2 = np.random.rand(N)
    x = (1 - r1) * 0.0 + r1 * (1 - r2) * 1.0 + r1 * r2 * 0.5
    y = (1 - r1) * 0.0 + r1 * (1 - r2) * 0.0 + r1 * r2 * (np.sqrt(3)/2)
    inits.append(np.column_stack((x, y)))

    best_points = None
    best_score = -1

    for init in inits:
        # Ensure init is inside
        valid_init = []
        for p in init:
            if point_in_triangle(p):
                valid_init.append(p)
            else:
                # Project to centroid
                centroid = (V1 + V2 + V3) / 3
                valid_init.append(centroid + 0.8 * (p - centroid))
        init_params = np.array(valid_init).flatten()
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-6})
            # Check if result is better than initial
            res_score = -res.fun if res.fun < 0 else 0
            init_score = min_triangle_area(np.array(valid_init))
            
            if res.success or res_score > init_score:
                res_points = res.x.reshape(-1, 2)
                # Check validity
                is_valid = all(point_in_triangle(p) for p in res_points)
                if is_valid:
                    score = min_triangle_area(res_points)
                    if score > best_score:
                        best_score = score
                        best_points = res_points
        except Exception:
            pass

    if best_points is None:
        # Fallback: Return the best initial config
        for init in inits:
            valid_init = []
            for p in init:
                if point_in_triangle(p):
                    valid_init.append(p)
                else:
                    centroid = (V1 + V2 + V3) / 3
                    valid_init.append(centroid + 0.8 * (p - centroid))
            pts = np.array(valid_init)
            score = min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts

    return best_points
# EVOLVE-BLOCK-END
