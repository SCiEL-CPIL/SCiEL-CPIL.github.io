# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

S3H = np.sqrt(3) / 2
TIME_LIMIT = 115  # seconds, safety margin

def _min_area_raw(points):
    """
    Compute the minimum triangle area among all triples of points.
    Vectorized for speed.
    """
    n = len(points)
    # Generate all unique pairs (i, j) with i < j
    # We need triples (i, j, k) with i < j < k
    # It's faster to compute areas for all i, then find min for each i, then global min?
    # Or just vectorize the cross product for all triples.
    
    # Number of triples is C(11, 3) = 165. This is small.
    # Simple loop is fine, but let's try to make it robust.
    
    min_a = float('inf')
    # Precompute coordinates
    xs = points[:, 0]
    ys = points[:, 1]
    
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorize k loop
            # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
            dx_j = xs[j] - xs[i]
            dy_j = ys[j] - ys[i]
            
            # k > j
            dx_k = xs[j+1:] - xs[i]
            dy_k = ys[j+1:] - ys[i]
            
            cross = np.abs(dx_j * dy_k - dy_j * dx_k) * 0.5
            if len(cross) > 0:
                local_min = np.min(cross)
                if local_min < min_a:
                    min_a = local_min
    return min_a

def _penalized_objective(x):
    """
    Objective function for Nelder-Mead.
    x: flat array of 2*N values. Pairs (u, v) are barycentric coords.
    Returns: -min_area + penalty.
    """
    u = x[0::2]
    v = x[1::2]
    
    # Penalty for invalid points (u>=0, v>=0, u+v<=1)
    # We want to push points into the triangle.
    # u, v, w = 1-u-v >= 0
    invalid_u = np.maximum(0, -u)
    invalid_v = np.maximum(0, -v)
    invalid_w = np.maximum(0, u + v - 1.0)
    
    # Quadratic penalty
    penalty = 100.0 * (np.sum(invalid_u**2) + np.sum(invalid_v**2) + np.sum(invalid_w**2))
    
    # Convert to Cartesian
    pts = np.zeros((N, 2))
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = S3H * v
    
    min_a = _min_area_raw(pts)
    
    return -min_a + penalty

def _is_valid(pts, tol=1e-6):
    """Check if points are inside the triangle."""
    px = pts[:, 0]
    py = pts[:, 1]
    # Barycentric coordinates
    w0 = 1.0 - px - py / S3H
    w1 = px - py / (2.0 * S3H)
    w2 = py / S3H
    return np.min(w0) >= -tol and np.min(w1) >= -tol and np.min(w2) >= -tol

def _clamp_to_triangle(u, v):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        u[mask] /= sum_uv[mask]
        v[mask] /= sum_uv[mask]
    return u, v

def _bary_to_cart(u, v):
    """Convert barycentric coords to Cartesian."""
    x = u + 0.5 * v
    y = S3H * v
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -float('inf')
    
    # Heuristic initializations
    # 1. Grid-like distribution
    # 2. Random
    # 3. Centroid + perturbations
    
    initializations = []
    
    # 1. Grid
    # 3x3 grid + 2 corners? No, 11 points.
    # Let's just use random starts and a few structured ones.
    
    # Structured: 4 corners/edges + 7 interior
    # Actually, for Heilbronn, spreading out is key.
    
    # Start 1: Random
    u1 = np.random.random(N)
    v1 = np.random.random(N)
    # Ensure valid by rejection or clamping
    # Clamping is safer for initialization
    u1, v1 = _clamp_to_triangle(u1, v1)
    initializations.append(np.concatenate([u1, v1]))
    
    # Start 2: Grid-ish
    # Place points in a grid pattern within the triangle
    u_grid = []
    v_grid = []
    # 3 rows
    # Row 1 (bottom): 4 points
    for i in range(4):
        u_grid.append(0.1 + 0.2*i)
        v_grid.append(0.1)
    # Row 2 (middle): 3 points
    for i in range(3):
        u_grid.append(0.2 + 0.2*i)
        v_grid.append(0.3)
    # Row 3 (top): 2 points
    for i in range(2):
        u_grid.append(0.3 + 0.2*i)
        v_grid.append(0.5)
    # Row 4 (apex): 2 points? 
    # Total 4+3+2 = 9. Need 2 more.
    # Add near apex
    u_grid.append(0.4)
    v_grid.append(0.7)
    u_grid.append(0.5)
    v_grid.append(0.8)
    
    u_grid = np.array(u_grid)
    v_grid = np.array(v_grid)
    # Clamp to be safe
    u_grid, v_grid = _clamp_to_triangle(u_grid, v_grid)
    initializations.append(np.concatenate([u_grid, v_grid]))
    
    # More random starts
    for _ in range(10):
        u_r = np.random.random(N)
        v_r = np.random.random(N)
        u_r, v_r = _clamp_to_triangle(u_r, v_r)
        initializations.append(np.concatenate([u_r, v_r]))
    
    # Run Nelder-Mead from each initialization
    # Nelder-Mead is local, so multiple starts are crucial.
    
    for x0 in initializations:
        if time.time() > start_time + TIME_LIMIT - 5:
            break
            
        try:
            # Adaptive Nelder-Mead is robust for non-smooth functions
            res = minimize(
                _penalized_objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.x is not None:
                u = res.x[0::2]
                v = res.x[1::2]
                
                # Clamp to ensure validity
                u, v = _clamp_to_triangle(u, v)
                pts = _bary_to_cart(u, v)
                
                if _is_valid(pts):
                    min_a = _min_area_raw(pts)
                    if min_a > best_score:
                        best_score = min_a
                        best_points = pts.copy()
                        
        except Exception:
            continue

    # If we still have time, do a few more random restarts
    # Or refine the best point found
    if best_points is not None and time.time() < start_time + TIME_LIMIT - 10:
        # Convert best to barycentric
        px = best_points[:, 0]
        py = best_points[:, 1]
        u = px - 0.5 * py / S3H
        v = py / S3H
        x_best = np.concatenate([u, v])
        
        # Try to polish with a different method or just NM again with tighter tol
        try:
            res = minimize(
                _penalized_objective, 
                x_best, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.x is not None:
                u = res.x[0::2]
                v = res.x[1::2]
                u, v = _clamp_to_triangle(u, v)
                pts = _bary_to_cart(u, v)
                if _is_valid(pts):
                    min_a = _min_area_raw(pts)
                    if min_a > best_score:
                        best_score = min_a
                        best_points = pts.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = np.random.random(), np.random.random()
                if u + v < 1.0:
                    break
            best_points[i, 0] = u + 0.5 * v
            best_points[i, 1] = S3H * v

    return best_points

# EVOLVE-BLOCK-END
