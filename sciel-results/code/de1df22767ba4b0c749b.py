"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
H = np.sqrt(3) / 2.0

# Vertices of equilateral triangle for barycentric conversion
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])

def _bary_to_cart(u, v):
    """Convert barycentric coordinates (u, v) to Cartesian. w = 1-u-v."""
    w = 1.0 - u - v
    # u*V[0] + v*V[1] + w*V[2]
    # V[0] = [0,0], V[1] = [1,0], V[2] = [0.5, h]
    # x = 0*u + 1*v + 0.5*w = v + 0.5*(1-u-v) = 0.5 - 0.5u + 0.5v
    # y = 0*u + 0*v + h*w = h*(1-u-v)
    x = 0.5 - 0.5 * u + 0.5 * v
    y = H * (1.0 - u - v)
    return np.column_stack([x, y])

def _min_area_cartesian(points):
    """Compute minimum area among all C(11,3)=165 triangles. Points are (N,2) Cartesian."""
    p1 = points[_IDX[:, 0]]
    p2 = points[_IDX[:, 1]]
    p3 = points[_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _soft_min_area_cartesian(points, beta):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    Points are (N,2) Cartesian.
    Returns the soft-min value.
    """
    p1 = points[_IDX[:, 0]]
    p2 = points[_IDX[:, 1]]
    p3 = points[_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # We want to maximize min(areas), so we minimize -min(areas).
    # Soft min of areas: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, shift by max area (which is effectively the min of -areas)
    # Let z = -beta * areas. We want -1/beta * log(sum(exp(z)))
    # z_max = max(z) = -beta * min(areas)
    z = -beta * areas
    z_max = np.max(z)
    if z_max > 700:
        return 1e10
    log_sum = z_max + np.log(np.sum(np.exp(z - z_max)))
    soft_min_val = -log_sum / beta
    return soft_min_val

def _objective_bary(x, beta, penalty_weight):
    """
    Objective function in barycentric coordinates.
    x: (2N,) array of (u_0, v_0, u_1, v_1, ...)
    """
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Penalty for u+v > 1 (outside triangle)
    viol = np.maximum(0.0, u + v - 1.0)
    penalty = penalty_weight * np.sum(viol**2)
    
    # Convert to Cartesian
    points = _bary_to_cart(u, v)
    
    # Calculate soft min area
    soft_min = _soft_min_area_cartesian(points, beta)
    
    return -soft_min + penalty

def _build_structured_init_bary():
    """
    Build a structured 11-point configuration in barycentric coordinates.
    """
    # Vertices: (1,0), (0,1), (0,0)
    # Midpoints: (0.5, 0.5), (0.5, 0), (0, 0.5)
    # Centroid: (1/3, 1/3)
    # Additional points for better distribution
    
    inits = []
    
    # 1. Corners and midpoints and centroid
    pts = [
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0),
        (0.5, 0.5), (0.5, 0.0), (0.0, 0.5),
        (1/3, 1/3)
    ]
    # Need 4 more. Let's add points along edges or interior
    # (0.2, 0.2), (0.8, 0.2) is invalid (sum 1.0 is on boundary, ok)
    # Let's use a grid
    grid_u = [0.1, 0.3, 0.5, 0.7, 0.9]
    grid_v = [0.1, 0.3, 0.5, 0.7, 0.9]
    candidates = []
    for u in grid_u:
        for v in grid_v:
            if u + v <= 1.0:
                candidates.append((u, v))
    
    # Select 4 from candidates that are far from existing points
    existing = np.array(pts)
    selected = []
    for c in candidates:
        dists = np.min(np.sqrt((existing[:,0]-c[0])**2 + (existing[:,1]-c[1])**2))
        selected.append((dists, c))
    selected.sort(key=lambda x: -x[0])
    for _, c in selected[:4]:
        pts.append(c)
        
    inits.append(np.array([p for pair in pts for p in pair]))
    
    # 2. Random init
    rng = np.random.RandomState(42)
    u = rng.random(N)
    v = rng.random(N) * (1.0 - u)
    inits.append(np.concatenate([u, v]))
    
    # 3. Another random init
    u = rng.random(N)
    v = rng.random(N) * (1.0 - u)
    inits.append(np.concatenate([u, v]))
    
    return inits

def _local_polish_bary(x, deadline):
    """
    Local refinement in barycentric space using L-BFGS-B with increasing beta.
    """
    x_current = x.copy()
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Stages: (beta, penalty_weight, maxiter)
    stages = [
        (50.0, 100.0, 100),
        (200.0, 1000.0, 150),
        (1000.0, 10000.0, 200),
        (5000.0, 100000.0, 300)
    ]
    
    for beta, pw, maxiter in stages:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _objective_bary, 
                x_current, 
                args=(beta, pw), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < _objective_bary(x_current, beta, pw):
                x_current = res.x
            elif res.success:
                x_current = res.x
        except Exception:
            pass
    
    # Final evaluation
    uv = x_current.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    points = _bary_to_cart(u, v)
    
    # Clamp to ensure validity
    points = np.clip(points, 0, 1)
    # Re-check barycentric validity
    # If a point is outside, it might have drifted. 
    # The optimizer should keep it inside due to bounds and penalty, but let's be safe.
    # Actually, if bounds are 0..1 for u and v, and penalty for u+v>1, 
    # it's possible u+v > 1 slightly. We should project back or just accept if penalty was high.
    # Better: check if all points are inside.
    
    # Check if valid
    h = H
    valid = True
    for p in points:
        x, y = p[0], p[1]
        if x < -1e-6 or x > 1+1e-6 or y < -1e-6 or y > h+1e-6:
            valid = False
        if y > 2*h*x + 1e-4:
            valid = False
        if y > 2*h*(1-x) + 1e-4:
            valid = False
            
    if not valid:
        # Fallback to previous best or just return what we have, hoping it's close
        # For now, just return, the caller will check.
        pass
        
    min_area = _min_area_cartesian(points)
    return x_current, points, min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_x = None
    best_points = None
    
    inits = _build_structured_init_bary()
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        x_current = init.copy()
        
        # Run local polish
        polished_x, polished_points, min_area = _local_polish_bary(x_current, deadline)
        
        if min_area > best_score:
            best_score = min_area
            best_x = polished_x
            best_points = polished_points
            
    # Final polish on the best result with higher precision
    if best_x is not None and time.time() < deadline - 5:
        # Run one more time with very high beta
        try:
            res = minimize(
                _objective_bary, 
                best_x, 
                args=(10000.0, 1000000.0), 
                method='L-BFGS-B', 
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < _objective_bary(best_x, 10000.0, 1000000.0):
                best_x = res.x
                uv = best_x.reshape(N, 2)
                u = uv[:, 0]
                v = uv[:, 1]
                best_points = _bary_to_cart(u, v)
                best_score = _min_area_cartesian(best_points)
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        u = np.random.rand(N)
        v = np.random.rand(N) * (1.0 - u)
        x_final = np.concatenate([u, v])
        best_points = _bary_to_cart(u, v)
    else:
        x_final = best_x
        best_points = _bary_to_cart(x_final.reshape(N, 2)[:, 0], x_final.reshape(N, 2)[:, 1])
        
    return best_points
# EVOLVE-BLOCK-END
