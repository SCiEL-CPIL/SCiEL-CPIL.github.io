"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import itertools

# Deadline guard
_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _remaining_time():
    return DEADLINE - (time.time() - _START_TIME) - SAFETY_MARGIN

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # Use vectorized cross product for speed
    # Points shape: (n, 2)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(points):
    """Check if all points are inside or on the boundary of the unit equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric coordinates must be non-negative
    h = np.sqrt(3) / 2
    for p in points:
        x, y = p
        # barycentric: y/h >= 0, 1-x-y/h >= 0, x+y/h >= 0
        b0 = y / h
        b1 = 1 - x - y / h
        b2 = x + y / h - 1  # wait, let's use standard barycentric
        # For triangle with vertices A=(0,0), B=(1,0), C=(0.5, h)
        # Point P = b0*A + b1*B + b2*C where b0+b1+b2=1
        # x = b1*1 + b2*0.5
        # y = b2*h
        # So b2 = y/h
        # b1 = x - 0.5*b2 = x - 0.5*y/h
        # b0 = 1 - b1 - b2
        b2 = y / h
        b1 = x - 0.5 * b2
        b0 = 1 - b1 - b2
        if b0 < -1e-10 or b1 < -1e-10 or b2 < -1e-10:
            return False
    return True

def _objective(x):
    """Objective function: negative of min triangle area (to minimize)."""
    points = x.reshape(N, 2)
    if not _is_inside_triangle(points):
        return 1e10
    return -_min_triangle_area(points)

def _constraints(x):
    """Constraints for SLSQP: all points inside triangle."""
    points = x.reshape(N, 2)
    h = np.sqrt(3) / 2
    cons = []
    for i in range(N):
        x_i, y_i = points[i]
        b2 = y_i / h
        b1 = x_i - 0.5 * b2
        b0 = 1 - b1 - b2
        cons.append({'type': 'ineq', 'fun': lambda b0=b0: b0})
        cons.append({'type': 'ineq', 'fun': lambda b1=b1: b1})
        cons.append({'type': 'ineq', 'fun': lambda b2=b2: b2})
    return cons

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Initial guess: hexagonal grid-like arrangement
    h = np.sqrt(3) / 2
    
    # Start with a reasonable symmetric configuration
    # Place points in a hexagonal pattern scaled to fit in the triangle
    # For 11 points, use a hexagonal close packing subset
    
    # Let's use a few different initializations and pick the best
    initializations = []
    
    # Init 1: Uniform grid
    np.random.seed(42)
    pts1 = []
    for i in range(4):
        for j in range(4):
            if len(pts1) < 11:
                u = (i + 0.5) / 4
                v = (j + 0.5) / 4
                if u + v < 1.0:
                    pts1.append([u + 0.5 * v, h * v])
    while len(pts1) < 11:
        u, v = np.random.random(2)
        if u + v < 0.95:
            pts1.append([u + 0.5 * v, h * v])
    initializations.append(np.array(pts1[:11]))
    
    # Init 2: Hexagonal pattern
    pts2 = []
    # Hexagonal lattice points
    hex_pts = [
        [0.5, 0.25 * h],
        [0.35, 0.45 * h],
        [0.65, 0.45 * h],
        [0.25, 0.65 * h],
        [0.5, 0.65 * h],
        [0.75, 0.65 * h],
        [0.15, 0.85 * h],
        [0.35, 0.85 * h],
        [0.5, 0.85 * h],
        [0.65, 0.85 * h],
        [0.85, 0.85 * h],
    ]
    # Transform to triangle coordinates
    for px, py in hex_pts:
        # py is in [0,1] range for the triangle height
        x = px + 0.5 * py
        y = h * py
        pts2.append([x, y])
    initializations.append(np.array(pts2))
    
    # Init 3: Random
    np.random.seed(123)
    pts3 = []
    while len(pts3) < 11:
        u, v = np.random.random(2)
        if u + v < 0.95:
            pts3.append([u + 0.5 * v, h * v])
    initializations.append(np.array(pts3))
    
    best_points = None
    best_score = -float('inf')
    
    for init_pts in initializations:
        if _remaining_time() < 20.0:
            break
            
        x0 = init_pts.flatten()
        
        # Use SLSQP to maximize min area
        try:
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                constraints=_constraints(x0),
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
        except Exception:
            continue
        
        if result.success or result.fun < 1e9:
            points = result.x.reshape(N, 2)
            if _is_inside_triangle(points):
                score = _min_triangle_area(points)
                if score > best_score:
                    best_score = score
                    best_points = points
    
    # If SLSQP didn't find good results, try simulated annealing
    if _remaining_time() > 10.0 and (best_points is None or best_score < 0.01):
        # Simulated annealing
        np.random.seed(7)
        if best_points is not None:
            current = best_points.copy()
        else:
            current = initializations[0].copy()
        
        current_score = _min_triangle_area(current) if _is_inside_triangle(current) else 0
        best_sa = current.copy()
        best_sa_score = current_score
        
        T = 0.01
        T_min = 1e-6
        cooling = 0.999
        
        for step in range(10000):
            if _remaining_time() < 5.0:
                break
                
            # Perturb
            new = current.copy()
            i = np.random.randint(N)
            new[i] += np.random.normal(0, T * 0.1, 2)
            
            # Check bounds
            if not _is_inside_triangle(new):
                continue
            
            new_score = _min_triangle_area(new)
            delta = new_score - current_score
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current = new
                current_score = new_score
                if current_score > best_sa_score:
                    best_sa_score = current_score
                    best_sa = current.copy()
            
            T *= cooling
            if T < T_min:
                break
        
        if best_sa_score > best_score:
            best_score = best_sa_score
            best_points = best_sa
    
    if best_points is None:
        # Fallback to initial program
        import random
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, h * v])
        best_points = np.array(points)
    
    return best_points
# EVOLVE-BLOCK-END
