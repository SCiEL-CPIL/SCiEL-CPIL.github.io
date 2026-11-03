import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_cluster_points(cluster, rotation_deg):
    """Get the 2D positions of hexagon centers for a given cluster and rotation."""
    pts = []
    for q, r in cluster:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        pts.append((x, y))
    theta = np.radians(rotation_deg)
    ct = np.cos(theta)
    st = np.sin(theta)
    rotated = []
    for x, y in pts:
        rx = x * ct - y * st
        ry = x * st + y * ct
        rotated.append((rx, ry))
    return rotated

def enclosing_hex_side(points, theta_deg, cx, cy):
    """Compute the minimum side length of a hexagon with orientation theta_deg, center (cx,cy) enclosing all points."""
    S = 0
    for px, py in points:
        dx = px - cx
        dy = py - cy
        r = np.sqrt(dx*dx + dy*dy)
        if r == 0:
            continue
        phi = np.arctan2(dy, dx)
        theta = np.radians(theta_deg)
        side_normals = theta + np.pi/6 + np.arange(6) * np.pi/3
        diffs = (phi - side_normals) % (2*np.pi)
        diffs = np.minimum(diffs, 2*np.pi - diffs)
        min_diff = np.min(diffs)
        if min_diff >= np.pi/2:
            return float('inf')
        S_i = (r * np.cos(min_diff)) / (np.sqrt(3)/2)
        S = max(S, S_i)
    return S

def solve():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Define some candidate cluster shapes in axial coordinates
    # Each is a list of (q, r) pairs
    clusters = []
    
    # Shape 1: 3x4 rectangle
    clusters.append([(q, r) for q in range(3) for r in range(4)])
    
    # Shape 2: 2x6 rectangle
    clusters.append([(q, r) for q in range(2) for r in range(6)])
    
    # Shape 3: 4x3 rectangle
    clusters.append([(q, r) for q in range(4) for r in range(3)])
    
    # Shape 4: Hexagonal shape (center + 6 around + 5 more)
    # Center: (0,0)
    # Ring 1: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Ring 2 partial: (2,0), (0,2), (-2,0), (0,-2), (2,-1), (-2,1), (1,1), (-1,-1)
    # Pick 5 from ring 2
    hex_shape = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), 
                 (2,0), (0,2), (-2,0), (0,-2), (2,-1)]
    clusters.append(hex_shape)
    
    # Shape 5: Another hexagonal arrangement
    hex_shape2 = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
                  (1,1), (-1,-1), (2,-1), (-2,1), (0,2)]
    clusters.append(hex_shape2)
    
    # Shape 6: Long strip with bends
    strip = [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0),
             (1,1), (2,1), (3,1), (4,1), (1,-1), (2,-1)]
    clusters.append(strip)
    
    best_S = float('inf')
    best_config = None
    
    rotations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    for cluster in clusters:
        if time.time() > deadline:
            break
        
        for rot in rotations:
            if time.time() > deadline:
                break
            
            pts_rot = get_cluster_points(cluster, rot)
            cx0 = np.mean([p[0] for p in pts_rot])
            cy0 = np.mean([p[1] for p in pts_rot])
            
            def objective(center):
                cx, cy = center
                return enclosing_hex_side(pts_rot, rot, cx, cy)
            
            best_local = None
            for trial in range(5):
                if trial == 0:
                    x0 = [cx0, cy0]
                elif trial == 1:
                    x0 = [cx0 + 1.0, cy0]
                elif trial == 2:
                    x0 = [cx0 - 1.0, cy0]
                elif trial == 3:
                    x0 = [cx0, cy0 + 1.0]
                else:
                    x0 = [cx0, cy0 - 1.0]
                
                try:
                    result = minimize(objective, x0, method='Nelder-Mead', 
                                     options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
                    if result.fun < (best_local[0] if best_local else float('inf')):
                        best_local = (result.fun, result.x)
                except:
                    pass
            
            if best_local and best_local[0] < best_S:
                best_S = best_local[0]
                best_config = (cluster, rot, best_local[1])
    
    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    cluster, rot, center = best_config
    cx, cy = center
    
    pts_rot = get_cluster_points(cluster, rot)
    
    inner = []
    for i, (px, py) in enumerate(pts_rot):
        hx = px + cx
        hy = py + cy
        inner.append([hx, hy, float(rot)])
    
    return inner, (float(cx), float(cy)), float(best_S), float(rot)

def construct():
    return solve()
