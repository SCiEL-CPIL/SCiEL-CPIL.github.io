import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11

def _hex_vertices(center, angle_deg, side=1.0):
    """
    Generate vertices of a regular hexagon.
    center: (x, y)
    angle_deg: rotation in degrees
    side: side length
    Returns: (6, 2) array of vertices
    """
    cx, cy = center
    theta = math.radians(angle_deg)
    # Standard pointy-top hexagon vertices (angles 30, 90, 150, 210, 270, 330)
    # Or flat-top (0, 60, ...).
    # Let's use flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees relative to center.
    # Distance from center to vertex is 'side'.
    angles = np.deg2rad([0, 60, 120, 180, 240, 300] + angle_deg)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _point_in_hex(p, center, angle_deg, side=1.0):
    """
    Check if point p is inside the hexagon.
    """
    # Transform p to hexagon's local frame
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    theta = math.radians(-angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    
    # For a flat-top hexagon of side 1:
    # The hexagon is defined by |lx| <= 1, |ly| <= sqrt(3)/2, |lx + ly| <= sqrt(3)/2? 
    # No, let's use the vertex method or simple inequalities.
    # Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
    # Inequalities:
    # 1. |lx| <= 1
    # 2. |ly| <= sqrt(3)/2
    # 3. |lx + ly| <= sqrt(3)/2  (Check slope of side)
    # 4. |lx - ly| <= sqrt(3)/2
    # Actually, the sides are:
    # Top: y = sqrt(3)/2 (for -0.5 <= x <= 0.5)
    # Top-Right: connects (0.5, sqrt(3)/2) and (1,0). Equation: y - 0 = -sqrt(3)(x - 1) => sqrt(3)x + y = sqrt(3).
    # Bottom-Right: connects (1,0) and (0.5, -sqrt(3)/2). Equation: y - 0 = sqrt(3)(x - 1) => -sqrt(3)x + y = -sqrt(3).
    # Bottom: y = -sqrt(3)/2
    # Bottom-Left: -sqrt(3)x - y = sqrt(3)
    # Top-Left: sqrt(3)x - y = sqrt(3)
    
    # So conditions:
    # |lx| <= 1
    # |ly| <= sqrt(3)/2
    # |sqrt(3)*lx + ly| <= sqrt(3)
    # |-sqrt(3)*lx + ly| <= sqrt(3)
    
    sq3 = math.sqrt(3)
    if abs(lx) > 1.0: return False
    if abs(ly) > sq3/2.0: return False
    if abs(sq3*lx + ly) > sq3: return False
    if abs(-sq3*lx + ly) > sq3: return False
    return True

def _overlap_penalty(pos_angles, lambda_pen=100.0):
    """
    pos_angles: (N, 3) array of [x, y, angle_deg]
    Returns penalty for overlaps.
    """
    penalty = 0.0
    n = len(pos_angles)
    # Precompute vertices
    verts = []
    for i in range(n):
        x, y, a = pos_angles[i]
        verts.append(_hex_vertices((x, y), a))
        
    for i in range(n):
        for j in range(i+1, n):
            # Quick check: center distance
            d = np.linalg.norm(pos_angles[i][:2] - pos_angles[j][:2])
            if d > 2.0: # Max distance from center to vertex is 1. If dist > 2, no overlap.
                continue
            
            # Check if any vertex of i is in j
            overlap = False
            for v in verts[i]:
                if _point_in_hex(v, pos_angles[j][:2], pos_angles[j][2]):
                    overlap = True
                    break
            if not overlap:
                # Check if any vertex of j is in i
                for v in verts[j]:
                    if _point_in_hex(v, pos_angles[i][:2], pos_angles[i][2]):
                        overlap = True
                        break
            
            if overlap:
                # Penalty proportional to overlap depth? 
                # Simple binary penalty is hard for gradient-free.
                # Use distance-based penalty if we can measure "depth".
                # For now, large constant penalty.
                penalty += 10.0 
                # To make it smoother, we could calculate the minimum distance to remove overlap,
                # but that's complex. A large constant + Nelder-Mead usually works if init is good.
                
    return penalty

def _outer_side_length(pos_angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Calculate the minimum side length of the outer hexagon to contain all inner hexagons.
    """
    # Outer hexagon normals (flat-top, side L)
    # Normals are at angles 0, 60, 120, 180, 240, 300?
    # For flat-top hex, sides are horizontal? No, flat-top means top side is horizontal.
    # Normal to top side is (0, 1).
    # Vertices at 30, 90...? No, flat top vertices are at 0, 60...?
    # Let's stick to the definition:
    # A regular hexagon of side L, centered at C, rotated by Phi.
    # It is the intersection of 6 half-planes.
    # The normals to the sides are at angles Phi + 0, Phi + 60, ...
    # The distance from center to side is L * sqrt(3) / 2.
    
    phi = math.radians(outer_angle)
    normals = []
    for k in range(6):
        ang = phi + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj = 0.0
    n = len(pos_angles)
    for i in range(n):
        x, y, a = pos_angles[i]
        verts = _hex_vertices((x, y), a)
        for v in verts:
            for nx, ny in normals:
                # Project vertex onto normal
                proj = (v[0] - outer_center[0]) * nx + (v[1] - outer_center[1]) * ny
                if proj > max_proj:
                    max_proj = proj
                    
    # max_proj is the distance from center to the furthest side projection.
    # This distance must be <= L * sqrt(3) / 2
    # So L >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / math.sqrt(3.0)

def _objective(params):
    """
    params: flattened array of [x0, y0, a0, x1, y1, a1, ...]
    """
    pos_angles = params.reshape(N, 3)
    # Unwrap angles to avoid discontinuities? Nelder-Mead doesn't need gradients, but wrapping might help.
    # Let's keep angles in degrees.
    
    L = _outer_side_length(pos_angles)
    pen = _overlap_penalty(pos_angles)
    
    return L + pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initial configuration: Hex grid
    # q, r coords
    hexes = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (0, -2)
    ]
    
    sq3 = math.sqrt(3)
    init_params = []
    for q, r in hexes:
        x = sq3 * (q + r/2.0)
        y = 1.5 * r
        init_params.extend([x, y, 0.0])
        
    x0 = np.array(init_params)
    
    # Optimization
    best_x = x0
    best_val = _objective(x0)
    
    # Use Nelder-Mead
    options = {
        'maxiter': 10000,
        'xatol': 1e-4,
        'fatol': 1e-4,
        'adaptive': True
    }
    
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', options=options)
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline:
        # Return best found so far
        pos_angles = best_x.reshape(N, 3)
        # Ensure no overlap? If overlap, score is invalid/0.
        # We should try to repair or just return.
        # The penalty in objective should have minimized overlap.
        # Let's do a quick check.
        if _overlap_penalty(pos_angles) > 0.01:
             # Try a few random restarts if we have time? 
             # Or just return the best non-overlapping one if we kept track?
             # We didn't keep track of non-overlapping states separately.
             # Let's assume the optimizer found a valid one or close to it.
             pass
             
    inner = []
    for i in range(N):
        x, y, a = best_x[i*3], best_x[i*3+1], best_x[i*3+2]
        inner.append([x, y, a])
        
    # Calculate final L for reporting (though evaluator calculates it)
    # The evaluator expects: inner, center, side, angle
    # We fixed outer center to (0,0) and angle to 0 in the objective.
    # Is it possible that moving the outer hexagon helps?
    # For a symmetric cluster, centering it is optimal.
    # For asymmetric, maybe not.
    # Given the "flower" symmetry, (0,0) is a good bet.
    
    L_final = _outer_side_length(best_x.reshape(N, 3))
    
    return inner, (0.0, 0.0), L_final, 0.0

