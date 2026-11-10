import numpy as np
from scipy.optimize import minimize
import time

def get_hex_vertices(cx, cy, s=1.0, angle_deg=0):
    """
    Returns vertices of a regular hexagon.
    """
    angles = np.deg2rad(angle_deg) + np.deg2rad(np.arange(0, 360, 60))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def min_hex_side_length(points, angle_deg=0.0):
    """
    Calculate the minimum side length of a regular hexagon (centered at origin, given angle)
    that contains all given points.
    """
    if len(points) == 0:
        return 0.0
    
    base_angle = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        angle = base_angle + np.deg2rad(60 * i)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    # Project points onto normals
    projections = points @ normals.T
    # We need max absolute projection
    max_proj = np.max(np.abs(projections))
    
    # Apothem = S * sqrt(3) / 2
    # So S = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def objective(params):
    """
    Objective function for optimization.
    params: array of shape (22,) representing 11 (x,y) pairs.
    """
    pts = params.reshape(11, 2)
    
    # Re-center to prevent drift
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    
    # Get all vertices of all hexagons
    all_vertices = []
    for i in range(11):
        verts = get_hex_vertices(pts[i, 0], pts[i, 1])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Calculate required side length
    side_len = min_hex_side_length(all_vertices)
    
    # Calculate overlap penalty
    penalty = 0.0
    min_dist = np.sqrt(3.0)
    for i in range(11):
        for j in range(i + 1, 11):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    # Scale penalty to be significant
    return side_len + 100.0 * penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    best_side = np.inf
    best_params = None
    best_pts = None
    
    # Generate several initial guesses
    inits = []
    
    # Init 1: 7-hex core + 4 hexes at 2nd ring (30, 150, 270, 30+180)
    base_pts = np.zeros((11, 2))
    base_pts[0] = [0, 0]
    for i in range(6):
        angle = np.deg2rad(60 * i)
        base_pts[i+1] = np.array([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
    for i in range(4):
        angle = np.deg2rad(30 + 90 * i)
        base_pts[7+i] = np.array([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
    inits.append(base_pts.flatten())
    
    # Init 2: Jittered version
    inits.append(base_pts.flatten() + np.random.normal(0, 0.5, 22))
    
    # Init 3: Hex grid axial coords
    hex_positions = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (1, 1), (-1, -1)
    ]
    pts3 = np.zeros((11, 2))
    for idx, (q, r) in enumerate(hex_positions):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        pts3[idx] = [x, y]
    inits.append(pts3.flatten())
    
    # Init 4: Ring of 6 + 4 at 0,90,180,270
    pts4 = np.zeros((11, 2))
    pts4[0] = [0, 0]
    for i in range(6):
        angle = np.deg2rad(60 * i)
        pts4[i+1] = np.array([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
    for i in range(4):
        angle = np.deg2rad(90 * i)
        pts4[7+i] = np.array([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
    inits.append(pts4.flatten())
    
    # Init 5: Random
    inits.append(np.random.normal(0, 3, 22))
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Try multiple optimization methods
        methods = ['Nelder-Mead', 'L-BFGS-B', 'Powell']
        
        for method in methods:
            if time.time() > deadline:
                break
            
            try:
                if method == 'Nelder-Mead':
                    res = minimize(objective, init, method=method,
                                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                elif method == 'L-BFGS-B':
                    res = minimize(objective, init, method=method,
                                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                else:  # Powell
                    res = minimize(objective, init, method=method,
                                   options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-10})
            except Exception:
                continue
            
            if time.time() > deadline:
                break
            
            if res.fun < best_side:
                params = res.x
                pts = params.reshape(11, 2)
                centroid = np.mean(pts, axis=0)
                pts = pts - centroid
                
                # Verify no overlap
                valid = True
                min_dist = np.sqrt(3.0)
                for i in range(11):
                    for j in range(i+1, 11):
                        dx = pts[i, 0] - pts[j, 0]
                        dy = pts[i, 1] - pts[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < min_dist - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    all_vertices = []
                    for i in range(11):
                        verts = get_hex_vertices(pts[i, 0], pts[i, 1])
                        all_vertices.append(verts)
                    all_vertices = np.vstack(all_vertices)
                    side_len = min_hex_side_length(all_vertices)
                    
                    if side_len < best_side:
                        best_side = side_len
                        best_params = params.copy()
                        best_pts = pts.copy()
    
    # Final polish: if we have a good solution, try to refine it
    if best_params is not None and time.time() < deadline - 30:
        try:
            res_final = minimize(objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res_final.fun < best_side:
                params = res_final.x
                pts = params.reshape(11, 2)
                centroid = np.mean(pts, axis=0)
                pts = pts - centroid
                
                valid = True
                min_dist = np.sqrt(3.0)
                for i in range(11):
                    for j in range(i+1, 11):
                        dx = pts[i, 0] - pts[j, 0]
                        dy = pts[i, 1] - pts[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < min_dist - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    all_vertices = []
                    for i in range(11):
                        verts = get_hex_vertices(pts[i, 0], pts[i, 1])
                        all_vertices.append(verts)
                    all_vertices = np.vstack(all_vertices)
                    side_len = min_hex_side_length(all_vertices)
                    
                    if side_len < best_side:
                        best_side = side_len
                        best_params = params.copy()
                        best_pts = pts.copy()
        except Exception:
            pass
    
    # Fallback
    if best_params is None:
        inner_hex_data = np.zeros((11, 3))
        for i in range(11):
            inner_hex_data[i, 0] = (i - 5) * np.sqrt(3)
            inner_hex_data[i, 1] = 0
            inner_hex_data[i, 2] = 0
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 2.0 * (5*np.sqrt(3) + 1) / np.sqrt(3) * (1 + 1e-6)
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Construct return values
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = best_pts[:, 0]
    inner_hex_data[:, 1] = best_pts[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0, 0, 0])
    
    outer_hex_side_length = best_side * (1 + 1e-6)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
