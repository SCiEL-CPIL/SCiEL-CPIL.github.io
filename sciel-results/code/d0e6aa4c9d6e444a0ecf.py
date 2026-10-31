import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
TIME_LIMIT = 115.0

def calc_min_area_bary(bary_points):
    """
    Calculate the minimum area (relative to total triangle area) of any triangle 
    formed by 3 of the given barycentric points.
    Input: (N, 3) array of barycentric coordinates.
    Output: float min area.
    """
    n = bary_points.shape[0]
    min_area = np.inf
    # Iterate over all combinations of 3 points
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = bary_points[i]
                p2 = bary_points[j]
                p3 = bary_points[k]
                # det of 3x3 matrix with rows p1, p2, p3
                # Since sum of coords is 1, we can use 2x2 det of differences for speed?
                # Area = 0.5 * | (p2-p1) x (p3-p1) | in 2D.
                # In barycentric, the area ratio is |det([p1, p2, p3])|.
                # Let's stick to det for correctness, it's 165 dets, very fast.
                det_val = np.linalg.det(np.array([p1, p2, p3]))
                area = abs(det_val)
                if area < min_area:
                    min_area = area
    return min_area

def bary_to_cart(bary):
    """Convert barycentric (u, v, w) to Cartesian (x, y) for triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = bary[:, 2]
    x = v + 0.5 * w
    y = w * (np.sqrt(3) / 2)
    return np.column_stack([x, y])

def flat_to_bary(flat):
    """Convert flat (u0, v0, u1, v1, ...) to (N, 3) barycentric, ensuring validity."""
    u = flat[0::2]
    v = flat[1::2]
    w = 1 - u - v
    
    # Clip to ensure points are inside the triangle
    # If a point goes outside, we project it back or just clip.
    # Clipping u, v, w to [0, 1] and normalizing is a standard robust approach.
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    
    s = u + v + w
    # Avoid division by zero
    mask = s > 0
    u = u.copy(); v = v.copy(); w = w.copy()
    u[mask] /= s[mask]
    v[mask] /= s[mask]
    w[mask] /= s[mask]
    
    # For points where s was 0 (should not happen after clip unless all 0, which is impossible if clipped from valid-ish inputs, but safe to handle)
    # If s==0, it means u=v=w=0, which is not a valid point. We can default to centroid.
    zero_mask = ~mask
    if np.any(zero_mask):
        u[zero_mask] = 1/3
        v[zero_mask] = 1/3
        w[zero_mask] = 1/3

    return np.column_stack([u, v, w])

def objective(flat):
    """
    Objective function for Nelder-Mead.
    Input: flat array of 22 floats (u0, v0, ..., u10, v10).
    Output: negative min area.
    """
    bary = flat_to_bary(flat)
    min_area = calc_min_area_bary(bary)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_score = -np.inf
    best_flat = None
    
    # Define several initial configurations
    inits = []
    
    # 1. Heuristic: Center + 6 around + 4 more
    # Centroid is (1/3, 1/3, 1/3) -> (1/3, 1/3) in (u,v)
    center = np.array([1/3, 1/3])
    
    # 6 points on the "hexagon" near the edges
    # Midpoints of edges in barycentric: (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)
    # Let's place 6 points in a ring around the center
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    radius = 0.2
    # Note: Barycentric space is not Euclidean, but we can approximate positions in (u,v)
    # (u,v) plane is a triangle. Center is (1/3, 1/3).
    # We'll just use random-ish or grid inits to be safe.
    
    # Init 1: Uniform Grid 3x3 clipped
    us = np.linspace(0.1, 0.8, 3)
    vs = np.linspace(0.1, 0.8, 3)
    grid_pts = []
    for u in us:
        for v in vs:
            if u + v <= 0.9: # inside
                grid_pts.append([u, v])
    # We need 11 points. 3x3 grid has 9 points inside roughly.
    # Add 2 more: Center and one corner-ish
    grid_pts.append([1/3, 1/3])
    grid_pts.append([0.2, 0.2])
    # Take first 11
    init_grid = np.array(grid_pts[:11]).flatten()
    inits.append(init_grid)
    
    # Init 2: Random
    np.random.seed(42)
    for _ in range(5):
        # Generate random points in triangle
        # u ~ U(0,1), v ~ U(0, 1-u)
        u = np.random.rand(N)
        v = np.random.rand(N) * (1 - u)
        flat = np.column_stack([u, v]).flatten()
        inits.append(flat)
        
    # Init 3: "Golden" distribution
    # Place points to maximize minimum distance initially (greedy)
    # This is complex, so let's use a structured pattern:
    # 1 center, 6 in inner ring, 4 in outer ring
    init_struct = []
    # Center
    init_struct.append([1/3, 1/3])
    # Inner ring (radius ~0.15 in cartesian approx, map to bary)
    # Just use fixed offsets in (u,v)
    offsets = [
        [0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1], [0.1, 0.1], [-0.1, -0.1]
    ]
    for off in offsets:
        p = np.array([1/3, 1/3]) + np.array(off)
        # Clip to valid range
        p[0] = np.clip(p[0], 0.01, 0.98)
        p[1] = np.clip(p[1], 0.01, 0.98 - p[0])
        init_struct.append(p)
    # Outer ring
    offsets2 = [
        [0.2, 0], [-0.2, 0], [0, 0.2], [0, -0.2]
    ]
    for off in offsets2:
        p = np.array([1/3, 1/3]) + np.array(off)
        p[0] = np.clip(p[0], 0.01, 0.98)
        p[1] = np.clip(p[1], 0.01, 0.98 - p[0])
        init_struct.append(p)
        
    init_struct_flat = np.array(init_struct[:11]).flatten()
    inits.append(init_struct_flat)
    
    # Run Nelder-Mead on each init
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_flat = res.x.copy()
                
        except Exception:
            pass
            
    # If we still have time, do a few more random restarts
    attempts = 0
    while time.time() < deadline - 10 and attempts < 20:
        u = np.random.rand(N)
        v = np.random.rand(N) * (1 - u)
        x0 = np.column_stack([u, v]).flatten()
        
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_flat = res.x.copy()
        except Exception:
            pass
        attempts += 1
        
    if best_flat is None:
        # Fallback to center
        best_flat = np.full(22, 1/3)
        
    # Convert best flat to barycentric and then Cartesian
    best_bary = flat_to_bary(best_flat)
    cart = bary_to_cart(best_bary)
    
    return cart
