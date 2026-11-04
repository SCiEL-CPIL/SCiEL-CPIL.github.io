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

def _sat_overlap_fast(v1, v2, n1, n2):
    """
    Returns the minimum separation distance between two hexagons.
    Positive if separated, negative if overlapping (penetration depth).
    """
    min_overlap = float('inf')
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
            
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return -min_overlap

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
            d = _sat_overlap_fast(verts[i], verts[j], norms[i], norms[j])
            if d < -tol:
                return False
    return True

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    d1 = sqrt3 + gap
    d2 = 2 * sqrt3 + gap
    
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _objective(p, t0, deadline, beta=100.0):
    """
    Objective function: Minimize outer side length + penalty for overlaps.
    p: [x1, y1, r1, ..., x12, y12, r12, ox, oy, oang]
    """
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
    
    # Calculate overlaps
    overlaps = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap_fast(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                overlaps.append(-d)
                
    if overlaps:
        p_arr = np.array(overlaps)
        # Soft max penalty to maintain differentiability
        scaled = 2.0 * p_arr / beta
        max_s = np.max(scaled)
        if max_s > 500:
            penalty = np.sum(p_arr**2)
        else:
            lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
            penalty = beta * lse
    else:
        penalty = 0.0

    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    
    return L + 100.0 * penalty

def _polish(params, t0, deadline):
    """
    Multi-stage polishing to ensure high precision.
    """
    if time.time() - t0 > deadline - 10:
        return params
    
    # Stage 1: L-BFGS-B with moderate penalty
    try:
        res = minimize(
            _objective,
            params,
            args=(t0, deadline - 10, 50.0),
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxfun': 5000
            }
        )
        if res.fun < _objective(params, t0, deadline, 50.0):
            params = res.x
    except Exception:
        pass
    
    # Stage 2: L-BFGS-B with higher penalty to enforce strict disjointness
    if time.time() - t0 < deadline - 5:
        try:
            res = minimize(
                _objective,
                params,
                args=(t0, deadline - 5, 200.0),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 5000
                }
            )
            if res.fun < _objective(params, t0, deadline, 200.0):
                params = res.x
        except Exception:
            pass
            
    # Stage 3: Final high-precision L-BFGS-B polish
    if time.time() - t0 < deadline - 2:
        try:
            res = minimize(
                _objective,
                params,
                args=(t0, deadline - 2, 1000.0),
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxfun': 10000
                }
            )
            if res.fun < _objective(params, t0, deadline, 1000.0):
                params = res.x
        except Exception:
            pass
    
    return params

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Generate diverse initial configurations
    # The hexagonal lattice is the densest, so we start with variations of it.
    configs = []
    gap = 0.05
    
    # Standard grids
    configs.append(_make_hex_grid(4, 3, gap=gap))
    configs.append(_make_hex_grid(3, 4, gap=gap))
    configs.append(_make_hex_grid(2, 6, gap=gap))
    configs.append(_make_hex_grid(6, 2, gap=gap))
    
    # Rotated grids
    configs.append(_make_hex_grid(4, 3, rot_offset=30, gap=gap))
    configs.append(_make_hex_grid(3, 4, rot_offset=30, gap=gap))
    
    # Ring structure (center + 6 + 5)
    configs.append(_make_ring(gap=gap))
    
    # Slightly perturbed ring to break symmetry
    np.random.seed(42)
    ring_perturbed = _make_ring(gap=gap)
    ring_perturbed[:, :2] += np.random.randn(12, 2) * 0.01
    configs.append(ring_perturbed)
    
    best_L = float('inf')
    best_params = None
    
    for cfg in configs:
        if time_left() < 15:
            break
            
        # Try a few outer rotations
        for oang in [0.0, 30.0, 15.0, 45.0]:
            if time_left() < 10:
                break
                
            init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, oang]))
            
            try:
                # Initial optimization
                res = minimize(
                    _objective,
                    init_params,
                    args=(t0, DEADLINE - 10, 50.0),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxfun': 2000
                    }
                )
                
                # Check validity
                inner = res.x[:36].reshape(12, 3)
                if _check_disjoint(inner, tol=1e-8):
                    ox = res.x[36]
                    oy = res.x[37]
                    oang_val = res.x[38]
                    all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
                    all_pts[:, 0] -= ox
                    all_pts[:, 1] -= oy
                    L = _outer_hex_side_fast(all_pts, oang_val)
                    
                    if L < best_L:
                        best_L = L
                        best_params = res.x.copy()
            except Exception:
                pass
                
    if best_params is not None and time_left() > 20:
        # Polish the best candidate
        best_params = _polish(best_params, t0, DEADLINE)
        
        # Final exact containment check and adjustment
        inner = best_params[:36].reshape(12, 3)
        if _check_disjoint(inner, tol=1e-9):
            ox = best_params[36]
            oy = best_params[37]
            oang_val = best_params[38]
            all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
            all_pts[:, 0] -= ox
            all_pts[:, 1] -= oy
            L = _outer_hex_side_fast(all_pts, oang_val)
            
            if L < best_L:
                best_L = L
        
        inner = best_params[:36].reshape(12, 3)
        if _check_disjoint(inner, tol=1e-9):
            ox = best_params[36]
            oy = best_params[37]
            oang_val = best_params[38]
            all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
            all_pts[:, 0] -= ox
            all_pts[:, 1] -= oy
            L = _outer_hex_side_fast(all_pts, oang_val)
            
            if L < best_L:
                best_L = L

    if best_params is None:
        # Fallback: return a valid, though not optimal, configuration
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
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
    
    # Final validity check
    if not _check_disjoint(inner, tol=1e-9):
        # If the optimizer failed to maintain disjointness, fall back to the initial valid config
        # This is rare with the penalty method, but good for safety.
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
    
    return inner, outer_center, outer_side, outer_angle
