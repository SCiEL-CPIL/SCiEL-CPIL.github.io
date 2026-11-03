import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 such triples
_triples = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_triples = np.array(_triples)
_N_TRIPLES = len(_triples)

def _min_triangle_area(pts):
    """
    pts: array of shape (N, 2) with Cartesian coordinates.
    Returns the minimum area of a triangle formed by any 3 points.
    Area = 0.5 * |det([p_i - p_k, p_j - p_k])|
    """
    # pts shape: (N, 2)
    # Extract points for each triple
    # p_i: (N_TRIPLES, 2), p_j: (N_TRIPLES, 2), p_k: (N_TRIPLES, 2)
    pi = pts[_triples[:, 0]]
    pj = pts[_triples[:, 1]]
    pk = pts[_triples[:, 2]]
    
    # Vectorized cross product (2D)
    # (pi - pk) x (pj - pk) = (pi_x - pk_x)*(pj_y - pk_y) - (pi_y - pk_y)*(pj_x - pk_x)
    dx1 = pi[:, 0] - pk[:, 0]
    dy1 = pi[:, 1] - pk[:, 1]
    dx2 = pj[:, 0] - pk[:, 0]
    dy2 = pj[:, 1] - pk[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _is_inside_triangle(pts):
    """
    Check if all points are inside or on the boundary of the equilateral triangle
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    # Vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, H])
    
    # Function to check if point P is inside triangle ABC
    # Using barycentric coordinates or cross products
    # For a point P, it is inside if it is on the same side of each edge as the opposite vertex.
    
    # Edge AB: y >= 0
    # Edge BC: (C-B) x (P-B) >= 0 ? 
    # Vector BC = (-0.5, H). Normal pointing inward: (H, 0.5)
    # Dot product of (P-B) with inward normal should be >= 0
    # Edge CA: (A-C) x (P-C) >= 0 ?
    # Vector CA = (-0.5, -H). Normal pointing inward: (-H, 0.5)
    
    # Let's use the standard barycentric check or simple inequalities.
    # Inequalities for equilateral triangle with base on x-axis:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (Line AC: y = sqrt(3)x) -> sqrt(3)x - y >= 0
    # 3. y <= -sqrt(3) * (x - 1) (Line BC: y = -sqrt(3)(x-1)) -> -sqrt(3)x - y + sqrt(3) >= 0
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    cond1 = y >= -1e-9
    cond2 = SQRT3 * x - y >= -1e-9
    cond3 = -SQRT3 * x - y + SQRT3 >= -1e-9
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def _penalty(pts):
    """
    Penalty for points outside the triangle.
    Returns a large value if any point is outside.
    """
    if _is_inside_triangle(pts):
        return 0.0
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Distance to edges
    d1 = y  # Distance to AB (y=0)
    d2 = SQRT3 * x - y  # Distance to AC (scaled)
    d3 = -SQRT3 * x - y + SQRT3  # Distance to BC (scaled)
    
    # Normalize distances to actual Euclidean distances
    # Edge lengths are 1.
    # Altitude is H.
    # d1 is actual distance.
    # d2 / sqrt(1 + 3) = d2 / 2 is actual distance to line AC?
    # Line AC: sqrt(3)x - y = 0. Distance = |sqrt(3)x - y| / 2.
    # Line BC: -sqrt(3)x - y + sqrt(3) = 0. Distance = |-sqrt(3)x - y + sqrt(3)| / 2.
    
    dist1 = np.minimum(0, d1)
    dist2 = np.minimum(0, d2) / 2.0
    dist3 = np.minimum(0, d3) / 2.0
    
    total_violation = np.sum(np.square(dist1)) + np.sum(np.square(dist2)) + np.sum(np.square(dist3))
    return total_violation * 1000.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def _is_time_up():
        return time.time() > deadline

    best_area = 0.0
    best_pts = None
    
    # Initial guesses
    # 1. Regular grid / structured
    # 2. Random
    
    # Strategy: Multi-restart Nelder-Mead
    # We maximize min_area, so we minimize -min_area + penalty
    
    def objective(pts_flat):
        if _is_time_up():
            return 1e9
        pts = pts_flat.reshape(N, 2)
        area = _min_triangle_area(pts)
        pen = _penalty(pts)
        return -area + pen
    
    # Generate several diverse initial configurations
    rng = np.random.default_rng(42)
    
    inits = []
    
    # Init 1: Centroid + rings
    # 1 center, 6 inner ring, 4 outer ring?
    # Or just structured grid
    
    # Let's use a few random starts and one structured start
    for i in range(10):
        if i == 0:
            # Structured: 3x3 grid mapped to triangle?
            # Simple: random but constrained
            pts = rng.random((N, 2))
            # Project to triangle?
            # Just use random points inside for start
            # To ensure they are inside, we can generate in barycentric
            bary = rng.dirichlet([1, 1, 1], size=N)
            x = bary[:, 1] + 0.5 * bary[:, 2]
            y = H * bary[:, 2]
            pts = np.column_stack([x, y])
        else:
            # Random inside
            bary = rng.dirichlet([1, 1, 1], size=N)
            x = bary[:, 1] + 0.5 * bary[:, 2]
            y = H * bary[:, 2]
            pts = np.column_stack([x, y])
        
        inits.append(pts)
        
        # Also try a "near optimal" guess based on known solutions?
        # The Heilbronn triangle problem for n=11 has known configurations.
        # We can try to perturb a symmetric configuration.
        
    # Add a symmetric configuration guess
    # 11 points: 
    # 1 at center
    # 6 at vertices of a hexagon?
    # 4 at midpoints of edges?
    # This is hard to guess exactly. Random starts + Nelder-Mead should work if the landscape is not too rugged.
    
    # Run Nelder-Mead for each init
    for init in inits:
        if _is_time_up():
            break
            
        # Flatten
        x0 = init.flatten()
        
        # Nelder-Mead options
        options = {
            'maxiter': 2000,
            'xatol': 1e-6,
            'fatol': 1e-6,
            'adaptive': True
        }
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options=options)
        except Exception:
            continue
            
        if _is_time_up():
            break
            
        res_pts = res.x.reshape(N, 2)
        
        # Check validity
        if not _is_inside_triangle(res_pts):
            # Project or discard?
            # If penalty was high, it might be outside.
            # Let's just calculate area, but if outside, it's invalid for final return.
            # However, Nelder-Mead might wander outside.
            # We can project back.
            pass
        
        area = _min_triangle_area(res_pts)
        
        # If points are outside, the area calculation is still valid mathematically,
        # but the constraint is violated.
        # We should only accept if inside.
        if _is_inside_triangle(res_pts):
            if area > best_area:
                best_area = area
                best_pts = res_pts.copy()
        
        # Also check the initial point
        init_area = _min_triangle_area(init)
        if _is_inside_triangle(init) and init_area > best_area:
            best_area = init_area
            best_pts = init.copy()

    # If we still have time, do a final polish on the best point found
    if best_pts is not None and not _is_time_up():
        x0 = best_pts.flatten()
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            res_pts = res.x.reshape(N, 2)
            if _is_inside_triangle(res_pts):
                area = _min_triangle_area(res_pts)
                if area > best_area:
                    best_area = area
                    best_pts = res_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback: return a valid configuration
        # e.g. all points at center (area 0) - bad, but valid
        # Better: random valid points
        bary = np.ones((N, 3)) / 3.0
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = H * bary[:, 2]
        best_pts = np.column_stack([x, y])
        
    return best_pts
