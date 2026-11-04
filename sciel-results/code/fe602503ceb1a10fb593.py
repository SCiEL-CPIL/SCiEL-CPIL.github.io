import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    # Flat-top hexagon: vertices at 0, 60, 120, 180, 240, 300
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    # Normals for flat-top hexagon: 30, 90, 150, 210, 270, 330
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap_fast(v1, v2, n1, n2):
    """
    Returns penetration depth if overlapping (negative), or separation distance if not.
    Uses precomputed normals for speed.
    """
    min_overlap = float('inf')
    
    # Check normals of hexagon 1
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            # Separated on this axis
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    # Check normals of hexagon 2
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
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    # Normals for the outer hexagon sides
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    # For a regular hexagon, distance from center to side (apothem) is side * sqrt(3)/2
    # So side = 2 / sqrt(3) * apothem
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
    """
    Create a hex grid of unit hexagons.
    For flat-top hexagons (vertices at 0, 60...):
    Horizontal spacing: 2 * cos(30) = sqrt(3)
    Vertical spacing: 1.5
    Offset for odd rows: sqrt(3)/2
    """
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
    # Center the configuration
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.0):
    """
    1 center + 6 ring + 5 outer.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    # Distance to touch center for flat-top: 2 * cos(30) = sqrt(3)
    d1 = sqrt3 + gap
    # Distance for outer ring: 2 * sqrt(3)
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

def _make_strip(n, gap=0.0):
    """
    A single row of hexagons.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    for i in range(n):
        pts.append([i * (sqrt3 + gap), 0, 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    return pts

def _objective(p, t0, deadline, beta=20.0):
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
    
    penalty = 0.0
    overlaps = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap_fast(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                overlaps.append(-d)
                
    if overlaps:
        p_arr = np.array(overlaps)
        # Use log-sum-exp for smooth max penalty
        scaled = p_arr / beta
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        penalty = beta * lse
    else:
        penalty = 0.0

    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    
    return L + 50.0 * penalty

def _polish(params, t0, deadline):
    """
    High-precision polish pass with multi-stage refinement.
    """
    if time.time() - t0 > deadline - 15:
        return params
    
    # Stage 1: L-BFGS-B with medium beta
    try:
        res = minimize(
            _objective,
            params,
            args=(t0, deadline - 10, 10.0),
            method='L-BFGS-B',
            options={
                'maxiter': 300,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'maxfun': 1500
            }
        )
        if res.fun < _objective(params, t0, deadline, 10.0):
            params = res.x
    except Exception:
        pass
    
    # Stage 2: L-BFGS-B with small beta for tighter convergence
    if time.time() - t0 > deadline - 8:
        return params
    
    try:
        res = minimize(
            _objective,
            params,
            args=(t0, deadline - 5, 5.0),
            method='L-BFGS-B',
            options={
                'maxiter': 200,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxfun': 1000
            }
        )
        if res.fun < _objective(params, t0, deadline, 5.0):
            params = res.x
    except Exception:
        pass
    
    # Stage 3: Final Nelder-Mead polish with very tight tolerances
    if time.time() - t0 > deadline - 5:
        return params
    
    try:
        res = minimize(
            _objective,
            params,
            args=(t0, deadline - 1, 2.0),
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-14,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        if res.fun < _objective(params, t0, deadline, 2.0):
            params = res.x
    except Exception:
        pass
        
    return params

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(_make_hex_grid(4, 3, gap=gap))
    configs.append(_make_hex_grid(3, 4, gap=gap))
    configs.append(_make_hex_grid(2, 6, gap=gap))
    configs.append(_make_hex_grid(6, 2, gap=gap))
    configs.append(_make_ring(gap=gap))
    configs.append(_make_hex_grid(4, 3, rot_offset=30, gap=gap))
    configs.append(_make_hex_grid(3, 4, rot_offset=30, gap=gap))
    configs.append(_make_hex_grid(2, 6, rot_offset=30, gap=gap))
    configs.append(_make_hex_grid(6, 2, rot_offset=30, gap=gap))
    
    # Add structured random starts with hexagonal symmetry hints
    np.random.seed(42)
    for _ in range(8):
        pts = np.random.randn(12, 3) * 1.5
        pts[:, 2] = np.random.uniform(0, 60, 12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)
    
    # Add a compact cluster configuration
    sqrt3 = np.sqrt(3)
    cluster_pts = []
    # Center
    cluster_pts.append([0, 0, 0.0])
    # First ring (6 hexagons at distance sqrt3)
    for k in range(6):
        ang = np.radians(60 * k)
        cluster_pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
    # Second ring partial (5 hexagons at distance 2*sqrt3)
    for k in range(5):
        ang = np.radians(60 * k + 30)
        cluster_pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
    cluster_pts = np.array(cluster_pts)
    cluster_pts[:, 0] -= np.mean(cluster_pts[:, 0])
    cluster_pts[:, 1] -= np.mean(cluster_pts[:, 1])
    configs.append(cluster_pts)
        
    best_L = float('inf')
    best_params = None
    
    # Optimization Loop - use larger beta for initial exploration
    for cfg in configs:
        if time_left() < 30:
            break
            
        for oang in [0.0, 30.0, 15.0, 45.0]:
            if time_left() < 20:
                break
                
            init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, oang]))
            
            # Try L-BFGS-B with larger beta for smooth exploration
            try:
                res = minimize(
                    _objective,
                    init_params,
                    args=(t0, DEADLINE - 25, 20.0),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-6,
                        'gtol': 1e-5,
                        'maxfun': 800
                    }
                )
                
                # Evaluate with smaller beta for comparison
                obj_val = _objective(res.x, t0, deadline, 5.0)
                if obj_val < _objective(init_params, t0, deadline, 5.0):
                    inner = res.x[:36].reshape(12, 3)
                    # Check validity
                    if _check_disjoint(inner):
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
                
    # Final refinement with high-precision polish if time permits
    if best_params is not None and time_left() > 30:
        best_params = _polish(best_params, t0, DEADLINE)
        
        # Re-evaluate the polished solution
        inner = best_params[:36].reshape(12, 3)
        if _check_disjoint(inner):
            ox = best_params[36]
            oy = best_params[37]
            oang_val = best_params[38]
            all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
            all_pts[:, 0] -= ox
            all_pts[:, 1] -= oy
            L = _outer_hex_side_fast(all_pts, oang_val)
            
            if L < best_L:
                best_L = L

    # Fallback if no valid solution found
    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    # Try a few more polish iterations if we have significant time left
    if time_left() > 15:
        for _ in range(2):
            if time_left() < 10:
                break
            try:
                best_params = _polish(best_params, t0, DEADLINE)
                inner = best_params[:36].reshape(12, 3)
                if _check_disjoint(inner):
                    ox = best_params[36]
                    oy = best_params[37]
                    oang_val = best_params[38]
                    all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
                    all_pts[:, 0] -= ox
                    all_pts[:, 1] -= oy
                    L = _outer_hex_side_fast(all_pts, oang_val)
                    if L < best_L:
                        best_L = L
            except Exception:
                break

    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
    # Final validity check
    if not _check_disjoint(inner):
        # If invalid, use fallback
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
    # Small safety margin
    outer_side *= 1.00001
    
    return inner, outer_center, outer_side, outer_angle
