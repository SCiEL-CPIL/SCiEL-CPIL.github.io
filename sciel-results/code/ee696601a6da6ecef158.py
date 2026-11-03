import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Vertices of the equilateral triangle
# The problem specifies vertices (0,0), (1,0), (1/2, sqrt(3)/2)
# Area = sqrt(3)/4
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, SQRT3 / 2.0])
CENTROID = (V1 + V2 + V3) / 3.0

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by N points.
    points: (N, 2) array.
    Returns: float, the minimum area.
    """
    n = len(points)
    min_area = np.inf
    
    # Vectorized calculation for speed
    # We need to compute areas for all triples (i, j, k) with i < j < k
    # Number of triples is C(11, 3) = 165. This is small enough for a loop,
    # but we can vectorize the cross product part if we generate indices.
    
    # For 11 points, a simple loop is very fast in Python if we avoid overhead.
    # Let's use a slightly optimized loop.
    for i in range(n - 2):
        pi = points[i]
        # Vector from i to all j > i
        # We need j > i and k > j
        # Let's just do the standard triple loop but keep it tight
        for j in range(i + 1, n - 1):
            pj = points[j]
            v_ij = pj - pi
            # Vector from i to all k > j
            # We can vectorize the cross product for fixed i, j
            # pk - pi for k in j+1..n-1
            pk_set = points[j+1:]
            v_ik_set = pk_set - pi
            # Cross product: v_ij[0]*v_ik[1] - v_ij[1]*v_ik[0]
            crosses = v_ij[0] * v_ik_set[:, 1] - v_ij[1] * v_ik_set[:, 0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _is_inside_triangle(p, v1, v2, v3, tol=1e-9):
    """Check if point p is inside triangle v1, v2, v3."""
    x, y = p
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    
    # Barycentric coordinates
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return False
        
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b
    
    return a >= -tol and b >= -tol and c >= -tol

def _objective_flat(x_flat):
    """
    Objective function for scipy.optimize.minimize.
    x_flat: 1D array of length 2*N (x1, y1, x2, y2, ...)
    Returns: -min_area + penalty
    """
    points = x_flat.reshape(N, 2)
    
    # Penalty for points outside the triangle
    # We use a large constant penalty to steer the optimizer inside
    penalty = 0.0
    for i in range(N):
        if not _is_inside_triangle(points[i], V1, V2, V3):
            penalty += 100.0 # Large penalty
            
    min_area = _min_triangle_area(points)
    
    # We want to maximize min_area, so we minimize -min_area
    return -min_area + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_points = None
    
    # Strategy: Multiple restarts with BFGS (Quasi-Newton)
    # BFGS is generally more robust than Nelder-Mead for smooth-ish functions
    # in higher dimensions, though the min-area function is non-smooth.
    # We mitigate this by using a sufficient number of restarts and tight tolerances.
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize points
        if restart == 0:
            # Structured initialization: Hexagonal packing approximation
            # Center
            pts = [CENTROID]
            # 6 points around center
            for k in range(6):
                angle = np.pi / 3 * k
                r = 0.2
                x = CENTROID[0] + r * np.cos(angle)
                y = CENTROID[1] + r * np.sin(angle)
                pts.append([x, y])
            # 4 more points near corners/edges
            # Corner 1: V1
            pts.append(V1 * 0.9) # Slightly inside
            # Corner 2: V2
            pts.append(V2 * 0.9)
            # Corner 3: V3
            pts.append(V3 * 0.9)
            # Mid-edge
            pts.append((V1 + V2) / 2.0 + np.array([0, 0.1]))
            
            points = np.array(pts)
        else:
            # Random initialization with rejection sampling
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    # Sample in bounding box
                    p = rng.uniform(0, 1, 2)
                    if _is_inside_triangle(p, V1, V2, V3):
                        points[i] = p
                        break
        
        # Flatten for optimizer
        x0 = points.flatten()
        
        # Define options for BFGS
        # BFGS requires gradient estimation. We let scipy handle it.
        # maxiter is limited to ensure we don't hang.
        # gtol: gradient norm tolerance.
        options = {
            'maxiter': 500,
            'gtol': 1e-6,
            'disp': False
        }
        
        try:
            res = minimize(_objective_flat, x0, method='BFGS', options=options)
            
            if res.success or res.fun < 10.0: # Allow some non-success if penalty is low
                new_points = res.x.reshape(N, 2)
                
                # Verify all points are inside
                valid = True
                for i in range(N):
                    if not _is_inside_triangle(new_points[i], V1, V2, V3, tol=1e-7):
                        valid = False
                        break
                
                if valid:
                    score = _min_triangle_area(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points.copy()
                        
        except Exception:
            pass

    # If no valid points found (should not happen), fallback
    if best_points is None:
        # Return centroid repeated? No, must be distinct.
        # Just return a random valid set
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                p = rng.uniform(0, 1, 2)
                if _is_inside_triangle(p, V1, V2, V3):
                    points[i] = p
                    break
        best_points = points
        
    return best_points
