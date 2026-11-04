import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # Safety margin below 150s

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_penetration(v1, v2, n1, n2):
    """
    Returns the maximum penetration depth (positive if overlapping, 0 if touching/separated).
    Uses SAT.
    """
    max_pen = 0.0
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0 # Separated
            
        pen = min(max1, max2) - max(min1, min2)
        if pen > max_pen:
            max_pen = pen
            
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0 # Separated
            
        pen = min(max1, max2) - max(min1, min2)
        if pen > max_pen:
            max_pen = pen
            
    return max_pen

def _outer_hex_side_fast(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner, tol=1e-9):
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
        
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_penetration(verts[i], verts[j], norms[i], norms[j])
            if d > tol:
                return False
    return True

def _make_flower_pattern(gap=0.0):
    """
    1 center, 6 in first ring, 5 in second ring.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    # Center
    pts.append([0, 0, 0.0])
    
    # First ring: distance sqrt(3)
    d1 = sqrt3 + gap
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        
    # Second ring: 5 hexagons. 
    # In a perfect hexagonal lattice, the second ring has 12. 
    # We take 5 of them, likely the ones at 30, 90, 150, 210, 270? 
    # Or maybe clustered? 
    # Let's place them at angles 0, 60, 120, 180, 240 but further out?
    # Actually, a good initial guess for 12 is just the first 12 points of a hexagonal grid.
    # Grid:
    #   . X X .
    #  X X X X
    #   . X X .
    # This is 2+4+2 = 8? No.
    # Let's stick to the grid generation which is safer.
    pass

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = 2.0 + gap
    spacing_y = sqrt3 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    # Center the grid
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _objective(p, t0, deadline, beta=100.0):
    if time.time() - t0 > deadline:
        return 1e6
    
    inner = p[:36].reshape(12, 3)
    ox = p[36]
    oy = p[37]
    oang = p[38]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    overlaps = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_penetration(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d > 0:
                overlaps.append(d)
                
    if overlaps:
        p_arr = np.array(overlaps)
        # Quadratic penalty is smooth enough for L-BFGS-B if we use a small enough step
        penalty = np.sum(p_arr**2) * beta
    else:
        penalty = 0.0

    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    
    return L + penalty

def _evaluate_config(inner, outer_center, outer_angle):
    if not _check_disjoint(inner, tol=1e-8):
        return False, float('inf')
    
    all_pts = np.vstack([_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(12)])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    L = _outer_hex_side_fast(all_pts, outer_angle)
    return True, L

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    configs = []
    
    # Grids
    configs.append(_make_hex_grid(4, 3, gap=0.1))
    configs.append(_make_hex_grid(3, 4, gap=0.1))
    configs.append(_make_hex_grid(2, 6, gap=0.1))
    configs.append(_make_hex_grid(6, 2, gap=0.1))
    
    # Rotated grids
    configs.append(_make_hex_grid(4, 3, rot_offset=30, gap=0.1))
    configs.append(_make_hex_grid(3, 4, rot_offset=30, gap=0.1))
    
    # Ring-like
    # 1 center, 6 around, 5 more
    pts = []
    sqrt3 = np.sqrt(3)
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
    # Add 5 more at distance 2*sqrt3?
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([2*sqrt3 * np.cos(ang), 2*sqrt3 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    configs.append(pts)
    
    best_L = float('inf')
    best_params = None
    
    # Optimize each config
    for idx, cfg in enumerate(configs):
        if time_left() < 10:
            break
            
        init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        
        # Try a few outer angles
        for oang in [0.0, 30.0]:
            if time_left() < 5:
                break
                
            init_params[38] = oang
            
            try:
                res = minimize(
                    _objective,
                    init_params,
                    args=(t0, DEADLINE - 10, 500.0),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxfun': 1000
                    }
                )
                
                inner = res.x[:36].reshape(12, 3)
                ox = res.x[36]
                oy = res.x[37]
                oang_val = res.x[38]
                
                valid, L = _evaluate_config(inner, (ox, oy), oang_val)
                if valid and L < best_L:
                    best_L = L
                    best_params = res.x.copy()
            except Exception:
                pass

    # Polish best
    if best_params is not None and time_left() > 20:
        try:
            res = minimize(
                _objective,
                best_params,
                args=(t0, DEADLINE - 5, 1000.0),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 5000
                }
            )
            inner = res.x[:36].reshape(12, 3)
            ox = res.x[36]
            oy = res.x[37]
            oang_val = res.x[38]
            valid, L = _evaluate_config(inner, (ox, oy), oang_val)
            if valid and L < best_L:
                best_L = L
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle

    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
    if not _check_disjoint(inner, tol=1e-9):
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle

    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle
