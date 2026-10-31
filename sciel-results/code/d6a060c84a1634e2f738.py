"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])
TRI_AREA = np.sqrt(3)/4

def compute_min_area(points):
    # points shape (N, 2)
    p1 = points[:, np.newaxis, np.newaxis, :]
    p2 = points[np.newaxis, :, np.newaxis, :]
    p3 = points[np.newaxis, np.newaxis, :, :]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    areas = 0.5 * np.abs(cross)
    
    i = np.arange(N)
    mask = (i[:, None, None] != i[None, :, None]) & \
           (i[None, :, None] != i[None, None, :]) & \
           (i[:, None, None] != i[None, None, :])
           
    areas[~mask] = np.inf
    
    return np.min(areas)

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    
    d1 = np.maximum(0, -y)
    d2 = np.maximum(0, y - np.sqrt(3) * x)
    d3 = np.maximum(0, np.sqrt(3) * x + y - np.sqrt(3))
    
    return np.sum(d1**2 + d2**2 + d3**2)

def objective(x):
    points = x.reshape(N, 2)
    min_area = compute_min_area(points)
    pen = penalty(points)
    return -min_area + 10.0 * pen

def generate_lattice_start():
    # 10 points from triangular grid side 4 + 1 center point
    pts = []
    for i in range(4):
        for j in range(4 - i):
            # Barycentric coords: u, v, w
            # u = 1 - (i+j)/3 ? No.
            # Map (i,j) to triangle.
            # i is step along V1, j is step along V2.
            # P = (1 - (i+j)/3) V0 + (i/3) V1 + (j/3) V2
            # But i+j can be up to 3.
            # If i=3, j=0: P = V1.
            # If i=0, j=3: P = V2.
            # If i=0, j=0: P = V0.
            u = 1.0 - (i + j) / 3.0
            v = i / 3.0
            w = j / 3.0
            p = u * V0 + v * V1 + w * V2
            pts.append(p)
    
    # Add 11th point: Centroid is already at i=1, j=1?
    # i=1, j=1: u=1/3, v=1/3, w=1/3. Yes.
    # So we need a different 11th point.
    # Add point at midpoint of edge V1-V2?
    # Midpoint of V1(1,0) and V2(0.5, sqrt(3)/2) is (0.75, sqrt(3)/4).
    pts.append((V1 + V2) / 2.0)
    
    return np.array(pts).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_x = None
    best_val = np.inf
    
    candidates = []
    candidates.append(generate_lattice_start())
    
    rng = np.random.RandomState(42)
    
    # Add some random starts
    for _ in range(20):
        x = rng.uniform(0, 1, N)
        y = rng.uniform(0, np.sqrt(3)/2, N)
        candidates.append(np.column_stack((x, y)).flatten())
    
    # Optimize each candidate
    for idx, x0 in enumerate(candidates):
        if time.time() > deadline:
            break
        
        try:
            # Use Nelder-Mead
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                
        except Exception:
            continue
            
    # If time permits, do more random restarts
    while time.time() < deadline - 5.0:
        x0 = rng.uniform(0, 1, N)
        y0 = rng.uniform(0, np.sqrt(3)/2, N)
        x0_flat = np.column_stack((x0, y0)).flatten()
        
        try:
            res = minimize(objective, x0_flat, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        best_x = generate_lattice_start()
        
    points = best_x.reshape(N, 2)
    return points
# EVOLVE-BLOCK-END
