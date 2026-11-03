import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# Reference triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Area of the reference triangle
AREA_REF = 0.5 * np.sqrt(3) / 2

def bary_to_cart(u, v):
    """Convert barycentric (u, v) to Cartesian coordinates. w = 1-u-v."""
    # P = u*V0 + v*V1 + w*V2
    # V0 is (0,0)
    # V1 is (1,0)
    # V2 is (0.5, sqrt(3)/2)
    # x = u*0 + v*1 + w*0.5 = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
    # y = u*0 + v*0 + w*sqrt(3)/2 = (1-u-v)*sqrt(3)/2
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = (np.sqrt(3) / 2.0) * w
    return x, y

def get_points(x):
    """Reshape flat vector x of length 2*N into N points (u, v)."""
    uv = x.reshape(N, 2)
    return uv

def compute_min_area(x):
    """
    Compute the minimum triangle area for points defined by barycentric coords x.
    Returns the raw area (not normalized).
    """
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Check constraints (soft penalty handled in objective)
    # Barycentric matrix B has rows [u_i, v_i, w_i]
    B = np.column_stack((u, v, w))
    
    min_area = np.inf
    # Iterate over all triplets
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                # Determinant of 3x3 matrix formed by barycentric coords
                # det(B[i], B[j], B[k])
                # This gives the ratio of areas.
                # We take absolute value.
                det_val = np.linalg.det(B[[i, j, k]])
                area = abs(det_val) * AREA_REF
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """
    Objective function to minimize.
    We want to maximize min_area, so we minimize -min_area.
    We add a penalty for points outside the triangle (w < 0).
    """
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Penalty for w < 0 (outside triangle)
    # Also penalty for u < 0 or v < 0, though bounds [0,1] should prevent most
    # But L-BFGS-B respects bounds, so u,v in [0,1].
    # The only constraint not enforced by bounds is u+v <= 1.
    viol = np.maximum(0.0, -w)
    penalty = 100.0 * np.sum(viol ** 2)
    
    min_area = compute_min_area(x)
    
    # If any point is outside, the area calculation might be weird or valid?
    # The determinant formula works for any affine combination, but geometric interpretation
    # as "area of triangle formed by points" assumes points are in the plane.
    # If a point is outside, the area is still well-defined.
    # However, the task requires points "on or inside".
    # The penalty ensures we stay inside.
    
    return -min_area + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_x = None
    best_score = -np.inf
    
    # Initializations
    # 1. Random points inside the triangle
    # 2. Symmetric points
    
    rng = np.random.default_rng(42)
    
    # Generate a set of initial candidates
    inits = []
    
    # Random starts
    for _ in range(50):
        u = rng.random(N)
        v = rng.random(N)
        # Ensure u+v <= 1 by scaling
        s = u + v
        s[s == 0] = 1e-9
        scale = np.minimum(1.0, 0.99 / s)
        u = u * scale
        v = v * scale
        x = np.concatenate([u, v])
        inits.append(x)
        
    # Structured start: Grid-like
    # 3x3 grid in barycentric coords?
    # Let's try a simple symmetric pattern
    # Center
    pts = []
    pts.append([0.5/3, 0.5/3]) # Centroid? No, bary (1/3, 1/3, 1/3) -> u=1/3, v=1/3
    # Vertices (slightly inside)
    eps = 0.05
    pts.append([eps, eps]) # Near V0 (0,0) -> u=0, v=0? No. V0 is (1,0,0) in bary?
    # My bary definition: P = u V0 + v V1 + w V2.
    # V0 = (1,0,0) -> u=1, v=0, w=0.
    # V1 = (0,1,0) -> u=0, v=1, w=0.
    # V2 = (0,0,1) -> u=0, v=0, w=1.
    
    # Let's re-verify barycentric mapping.
    # If P = V0, then u=1, v=0, w=0.
    # If P = V1, then u=0, v=1, w=0.
    # If P = V2, then u=0, v=0, w=1.
    
    # My previous random init:
    # u, v random. w = 1-u-v.
    # If u=1, v=0 -> P=V0.
    # If u=0, v=1 -> P=V1.
    # If u=0, v=0 -> P=V2.
    
    # So "inside" means u>=0, v>=0, u+v<=1.
    
    # Let's create a grid of points in the (u,v) plane within the triangle.
    # 11 points.
    # Maybe 1 center, 6 mid-edges, 3 vertices?
    # Vertices: (1,0), (0,1), (0,0)
    # Mid-edges: (0.5, 0.5), (0.5, 0), (0, 0.5)
    # Center: (1/3, 1/3)
    # That's 7. Need 4 more.
    # Mid-points of the segments from vertices to center?
    # (2/3, 1/6), (1/6, 2/3), (1/6, 1/6)
    # That's 10. One more?
    # Maybe adjust slightly.
    
    grid_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [0.5, 0.5], [0.5, 0.0], [0.0, 0.5],
        [1/3, 1/3],
        [2/3, 1/6], [1/6, 2/3], [1/6, 1/6]
    ]
    # Add one more random or specific point
    grid_pts.append([0.4, 0.4])
    
    x_grid = np.array(grid_pts).flatten()
    inits.append(x_grid)
    
    # Another structured start: Randomized grid
    for _ in range(5):
        x_g = x_grid + rng.normal(0, 0.05, size=x_grid.size)
        # Project back to feasible region?
        # Just clip u,v to [0,1] and ensure u+v<=1
        uv = x_g.reshape(N, 2)
        u = np.clip(uv[:, 0], 0, 1)
        v = np.clip(uv[:, 1], 0, 1)
        s = u + v
        scale = np.minimum(1.0, 0.99 / s)
        u = u * scale
        v = v * scale
        inits.append(np.concatenate([u, v]))
        
    # Optimization loop
    bounds = [(0.0, 1.0)] * (2 * N)
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.success or res.fun < best_score:
                score = -res.fun
                # Check validity
                uv = res.x.reshape(N, 2)
                u = uv[:, 0]
                v = uv[:, 1]
                w = 1 - u - v
                if np.all(w >= -1e-6) and np.all(u >= -1e-6) and np.all(v >= -1e-6):
                    # Clip to be safe
                    u = np.clip(u, 0, 1)
                    v = np.clip(v, 0, 1)
                    w = 1 - u - v
                    if np.all(w >= -1e-9):
                        best_x = res.x
                        best_score = score
        except Exception:
            pass
            
    if best_x is None:
        # Fallback to a simple valid configuration
        # Return the grid points
        best_x = x_grid
        
    # Convert best barycentric coords to Cartesian
    uv = best_x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Ensure strict validity
    w = 1.0 - u - v
    # If any w < 0, it's invalid. We should have avoided this, but just in case:
    # We can project points back into the triangle.
    # Simple projection: if w < 0, scale u,v such that u'+v'=1.
    mask = w < 0
    if np.any(mask):
        s = u[mask] + v[mask]
        s[s == 0] = 1e-9
        scale = 1.0 / s
        u[mask] = u[mask] * scale
        v[mask] = v[mask] * scale
        
    # Calculate final Cartesian coordinates
    # x = v + 0.5 * (1-u-v)
    # y = (sqrt(3)/2) * (1-u-v)
    w = 1.0 - u - v
    x_cart = v + 0.5 * w
    y_cart = (np.sqrt(3) / 2.0) * w
    
    points = np.column_stack((x_cart, y_cart))
    
    return points
