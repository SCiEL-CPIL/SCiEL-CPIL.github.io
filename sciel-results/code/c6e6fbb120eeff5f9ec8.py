import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap_fast(v1, v2, n1, n2):
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
    if len(pts) > 12:
        pts = pts[:12]
    elif len(pts) < 12:
        while len(pts) < 12:
            pts.append([0, 0, rot_offset])
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

def _make_packed_cluster(gap=0.0):
    """
    Create a tightly packed cluster of 12 hexagons.
    Uses a 3-3-3-3 arrangement with alternating offsets.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    
    # Row 0: 3 hexagons
    for c in range(3):
        x = c * spacing_x
        y = 0
        pts.append([x, y, 0.0])
    
    # Row 1: 3 hexagons, offset by half spacing
    for c in range(3):
        x = c * spacing_x + spacing_x / 2.0
        y = spacing_y
        pts.append([x, y, 0.0])
    
    # Row 2: 3 hexagons
    for c in range(3):
        x = c * spacing_x
        y = 2 * spacing_y
        pts.append([x, y, 0.0])
    
    # Row 3: 3 hexagons, offset by half spacing
    for c in range(3):
        x = c * spacing_x + spacing_x / 2.0
        y = 3 * spacing_y
        pts.append([x, y, 0.0])
        
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_strip(gap=0.0):
    """
    Create a strip-like arrangement of 12 hexagons.
    Two rows of 6 each.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    
    # Row 0: 6 hexagons
    for c in range(6):
        x = c * spacing_x
        y = 0
        pts.append([x, y, 0.0])
    
    # Row 1: 6 hexagons, offset by half spacing
    for c in range(6):
        x = c * spacing_x + spacing_x / 2.0
        y = spacing_y
        pts.append([x, y, 0.0])
        
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _evaluate_config(inner, outer_center, outer_angle):
    """
    Evaluate a configuration: return (L, is_valid).
    L is the outer side length, is_valid checks disjointness.
    """
    if not _check_disjoint(inner):
        return float('inf'), False
    
    all_pts = np.vstack([_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(12)])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    L = _outer_hex_side_fast(all_pts, outer_angle)
    return L, True

def _physics_relax(params, t0, deadline, iterations=500, damping=0.5):
    """
    Simple physics-based relaxation: push overlapping hexagons apart.
    params: [x1, y1, r1, ..., x12, y12, r12, ox, oy, oang]
    """
    params = params.copy()
    inner = params[:36].reshape(12, 3)
    
    for it in range(iterations):
        if time.time() - t0 > deadline:
            break
            
        # Compute forces
        forces = np.zeros((12, 2))
        
        for i in range(12):
            for j in range(i + 1, 12):
                v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                n1 = _hex_edge_normals(inner[i][2])
                n2 = _hex_edge_normals(inner[j][2])
                
                d = _sat_overlap_fast(v1, v2, n1, n2)
                
                if d < 0:
                    # Overlap: push apart
                    overlap = -d
                    # Direction: from center of j to center of i
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 1e-9:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Force magnitude proportional to overlap
                    f = overlap * 0.1
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
        
        # Update positions
        for i in range(12):
            inner[i][0] += forces[i, 0] * damping
            inner[i][1] += forces[i, 1] * damping
        
        params[:36] = inner.flatten()
    
    return params

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
    
    # Stage 3: Physics relaxation for final local refinement if time permits
    if time.time() - t0 < deadline - 2:
        try:
            params = _physics_relax(params, t0, deadline - 2, iterations=200)
        except Exception:
            pass
            
    return params

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    np.random.seed(42)
    
    best_L = float('inf')
    best_params = None
    
    # Initial configurations
    init_configs = []
    
    # 1. Hex Grid (standard packing)
    grid = _make_hex_grid(4, 3, gap=0.1)
    init_configs.append(np.concatenate((grid.flatten(), [0.0, 0.0, 0.0])))
    
    # 2. Ring (center + 6 + 5)
    ring = _make_ring(gap=0.1)
    init_configs.append(np.concatenate((ring.flatten(), [0.0, 0.0, 0.0])))
    
    # 3. Packed Cluster (3-3-3-3)
    cluster = _make_packed_cluster(gap=0.1)
    init_configs.append(np.concatenate((cluster.flatten(), [0.0, 0.0, 0.0])))
    
    # 4. Strip (2 rows of 6)
    strip = _make_strip(gap=0.1)
    init_configs.append(np.concatenate((strip.flatten(), [0.0, 0.0, 0.0])))
    
    # 5. Random restarts
    for _ in range(10):
        p = np.random.randn(36) * 2.0
        # Randomize rotations slightly
        p[2::3] = np.random.uniform(-15, 15, size=12)
        # Outer hex centered at origin initially
        p = np.concatenate((p, [0.0, 0.0, 0.0]))
        init_configs.append(p)
        
    # 6. Perturbed Grid
    for _ in range(5):
        grid = _make_hex_grid(4, 3, gap=0.0)
        p = grid.flatten() + np.random.randn(36) * 0.5
        p[2::3] = 0.0 # Keep rotations 0 for grid-like
        p = np.concatenate((p, [0.0, 0.0, 0.0]))
        init_configs.append(p)
        
    # 7. Perturbed Cluster
    for _ in range(5):
        cluster = _make_packed_cluster(gap=0.0)
        p = cluster.flatten() + np.random.randn(36) * 0.5
        p[2::3] = 0.0
        p = np.concatenate((p, [0.0, 0.0, 0.0]))
        init_configs.append(p)

    for cfg in init_configs:
        if time_left() < 10:
            break
            
        try:
            # Phase 1: Physics relaxation to quickly resolve overlaps
            if time_left() > 30:
                cfg = _physics_relax(cfg, t0, DEADLINE - 30, iterations=100)
            
            # Phase 2: Global-ish optimization with Nelder-Mead (robust to non-smoothness)
            if time_left() > 25:
                res = minimize(
                    _objective,
                    cfg,
                    args=(t0, DEADLINE - 25, 100.0),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                cfg = res.x
            
            # Phase 3: Polish with L-BFGS-B (fast local convergence)
            if time_left() > 15:
                cfg = _polish(cfg, t0, DEADLINE - 5)
            
            # Verify and update best
            if time_left() > 5:
                inner = cfg[:36].reshape(12, 3)
                ox = cfg[36]
                oy = cfg[37]
                oang = cfg[38]
                
                L, valid = _evaluate_config(inner, (ox, oy), oang)
                
                if valid and L < best_L:
                    best_L = L
                    best_params = cfg.copy()
                    
        except Exception:
            pass

    if best_params is None:
        # Fallback
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle

    inner = best_params[:36].reshape(12, 3)
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
    # Final check and safety margin
    L, valid = _evaluate_config(inner, outer_center, outer_angle)
    
    if not valid:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    all_pts = np.vstack([_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(12)])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    outer_side *= 1.0001
    
    return [[float(x) for x in row] for row in inner], outer_center, outer_side, outer_angle
