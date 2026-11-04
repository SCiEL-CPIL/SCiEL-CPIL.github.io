"""Optimize 12 unit hexagons inside a regular hexagon using SAT-based collision detection and gradient-free optimization."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
TIME_LIMIT = 145.0
SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, scale=1.0):
    """Return vertices of a regular hexagon (flat-top) centered at (cx, cy)."""
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + np.deg2rad(angle_deg)
    xs = cx + scale * np.cos(angles)
    ys = cy + scale * np.sin(angles)
    return np.column_stack((xs, ys))

def _hex_edge_normals(vertices):
    """Return outward unit normals for each edge of the hexagon."""
    n = len(vertices)
    normals = []
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge = v2 - v1
        # Outward normal (perpendicular to edge, pointing away from center)
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        # Ensure it points outward (away from center)
        center = np.mean(vertices, axis=0)
        midpoint = (v1 + v2) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal
        normals.append(normal)
    return np.array(normals)

def _sat_overlap_fast(v1, v2, n1, n2):
    """
    Fast SAT check: returns True if two convex polygons overlap.
    v1, v2: vertex arrays (6x2), n1, n2: normal arrays (6x2)
    """
    # Check separation along all normals of both polygons
    for i in range(6):
        nx, ny = n1[i]
        # Project v1
        proj1 = v1 @ np.array([nx, ny])
        min1, max1 = np.min(proj1), np.max(proj1)
        # Project v2
        proj2 = v2 @ np.array([nx, ny])
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 <= min2 + 1e-9 or max2 <= min1 + 1e-9:
            return False  # Separating axis found
    for i in range(6):
        nx, ny = n2[i]
        proj1 = v1 @ np.array([nx, ny])
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ np.array([nx, ny])
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 <= min2 + 1e-9 or max2 <= min1 + 1e-9:
            return False
    return True

def _outer_hex_side_fast(cx_arr, cy_arr, angle_arr, ocx, ocy, oangle_deg):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    apothem_unit = SQRT3 / 2.0
    oa = math.radians(oangle_deg)
    max_need = 0.0
    for k in range(6):
        na = oa + math.radians(90 + 60 * k)
        nx, ny = math.cos(na), math.sin(na)
        for i in range(N):
            v = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i], 1.0)
            proj = v @ np.array([nx, ny])
            max_proj = np.max(proj)
            val = (max_proj - (nx * ocx + ny * ocy)) / apothem_unit
            if val > max_need:
                max_need = val
    return max_need

def _check_disjoint(cx_arr, cy_arr, angle_arr):
    """Check if all hexagons are disjoint using SAT."""
    for i in range(N):
        v1 = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i], 1.0)
        n1 = _hex_edge_normals(v1)
        for j in range(i + 1, N):
            v2 = _hex_vertices(cx_arr[j], cy_arr[j], angle_arr[j], 1.0)
            n2 = _hex_edge_normals(v2)
            if _sat_overlap_fast(v1, v2, n1, n2):
                return False
    return True

def _make_hex_grid():
    """Create a 3x4 grid of hexagons."""
    pts = []
    rh = SQRT3
    cw = 2.0
    for row in range(3):
        for col in range(4):
            x = col * cw - 1.5 * cw
            y = row * 2 * rh - rh
            if row % 2 == 1:
                x += cw / 2
            pts.append([x, y, 0.0])
    return pts[:12]

def _make_ring():
    """Create a ring of 12 hexagons."""
    pts = []
    r = 2.5 * SQRT3
    for i in range(12):
        angle = math.radians(30 * i)
        pts.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    return pts

def _make_packed_cluster():
    """Create a flower-like cluster: center + 6 around + 5 more."""
    pts = [[0.0, 0.0, 0.0]]
    r1 = 2 * SQRT3
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle), 0.0])
    r2 = r1 * 1.732
    for deg in [30, 90, 150, 210, 270]:
        rad = math.radians(deg)
        pts.append([r2 * math.cos(rad), r2 * math.sin(rad), 0.0])
    return pts[:12]

def _make_strip():
    """Create a 2x6 strip."""
    pts = []
    rh = SQRT3
    cw = 2.0
    for row in range(2):
        for col in range(6):
            x = col * cw - 2.5 * cw
            y = row * 2 * rh - rh
            if row % 2 == 1:
                x += cw / 2
            pts.append([x, y, 0.0])
    return pts[:12]

def _evaluate_config(cfg, t0):
    """Evaluate a configuration: check validity and compute outer side."""
    if time.time() - t0 > TIME_LIMIT - 10:
        return None
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    if not _check_disjoint(cx_arr, cy_arr, angle_arr):
        return None
    
    best_side = float('inf')
    best_ocx, best_ocy, best_oa = 0.0, 0.0, 0.0
    for oa in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
        for ocx in np.linspace(-1, 1, 5):
            for ocy in np.linspace(-1, 1, 5):
                s = _outer_hex_side_fast(cx_arr, cy_arr, angle_arr, float(ocx), float(ocy), float(oa))
                if s < best_side:
                    best_side = s
                    best_ocx, best_ocy, best_oa = float(ocx), float(ocy), float(oa)
    return (best_side, best_ocx, best_ocy, best_oa, cx_arr, cy_arr, angle_arr)

def _objective(params, ocx, ocy, oangle_deg, beta=100.0):
    """Objective: outer side + penalty for overlaps."""
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    side = _outer_hex_side_fast(cx, cy, ang, ocx, ocy, oangle_deg)
    
    # Smooth overlap penalty using SAT
    penalty = 0.0
    for i in range(N):
        v1 = _hex_vertices(cx[i], cy[i], ang[i], 1.0)
        n1 = _hex_edge_normals(v1)
        for j in range(i + 1, N):
            v2 = _hex_vertices(cx[j], cy[j], ang[j], 1.0)
            n2 = _hex_edge_normals(v2)
            # Approximate overlap depth
            min_sep = float('inf')
            for k in range(6):
                nx, ny = n1[k]
                proj1 = v1 @ np.array([nx, ny])
                proj2 = v2 @ np.array([nx, ny])
                sep = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
                if sep < min_sep:
                    min_sep = sep
            for k in range(6):
                nx, ny = n2[k]
                proj1 = v1 @ np.array([nx, ny])
                proj2 = v2 @ np.array([nx, ny])
                sep = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
                if sep < min_sep:
                    min_sep = sep
            if min_sep < 0.05:
                penalty += (0.05 - min_sep) ** 2
    return side + beta * penalty

def _polish(cfg, t0, max_iter=200):
    """Polish a configuration using Nelder-Mead."""
    if time.time() - t0 > TIME_LIMIT - 15:
        return None
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    # Find best outer for this config
    best_side = float('inf')
    best_ocx, best_ocy, best_oa = 0.0, 0.0, 0.0
    for oa in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
        for ocx in np.linspace(-1, 1, 5):
            for ocy in np.linspace(-1, 1, 5):
                s = _outer_hex_side_fast(cx_arr, cy_arr, angle_arr, float(ocx), float(ocy), float(oa))
                if s < best_side:
                    best_side = s
                    best_ocx, best_ocy, best_oa = float(ocx), float(ocy), float(oa)
    
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = cx_arr[i]
        x0[3*i+1] = cy_arr[i]
        x0[3*i+2] = angle_arr[i]
    
    try:
        result = minimize(
            lambda p: _objective(p, best_ocx, best_ocy, best_oa, beta=150.0),
            x0, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        x_opt = result.x
        cx_opt = x_opt[0::3]
        cy_opt = x_opt[1::3]
        ang_opt = x_opt[2::3]
        
        if _check_disjoint(cx_opt, cy_opt, ang_opt):
            side, ocx, ocy, oa, _, _, _ = _evaluate_config(
                [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)], t0
            )
            if side is not None:
                return (side, ocx, ocy, oa, cx_opt, cy_opt, ang_opt)
    except Exception:
        pass
    return None

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Generate initial configs
    configs = [
        _make_hex_grid(),
        _make_ring(),
        _make_packed_cluster(),
        _make_strip(),
    ]
    
    # Add random configs
    for _ in range(3):
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 60)])
        configs.append(pts)
    
    # Phase 1: Evaluate initial configs
    for cfg in configs:
        if time.time() - t0 > TIME_LIMIT - 80:
            break
        result = _evaluate_config(cfg, t0)
        if result is not None:
            side, ocx, ocy, oa, cx_arr, cy_arr, angle_arr = result
            if side < best_side:
                best_side = side
                best_cfg = [[float(cx_arr[i]), float(cy_arr[i]), float(angle_arr[i])] for i in range(N)]
                best_center = (ocx, ocy)
                best_angle = oa
    
    if best_cfg is None:
        # Fallback to first config
        best_cfg = configs[0]
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        result = _evaluate_config(best_cfg, t0)
        if result is not None:
            best_side, best_center[0], best_center[1], best_angle = result[0], result[1], result[2], result[3]
    
    # Phase 2: Differential Evolution
    bounds = []
    for i in range(N):
        bounds.append((-4.0, 4.0))
        bounds.append((-4.0, 4.0))
        bounds.append((0.0, 60.0))
    
    def de_obj(params):
        if time.time() - t0 > TIME_LIMIT - 30:
            return 1e6
        return _objective(params, best_center[0], best_center[1], best_angle, beta=50.0)
    
    for start_idx in range(min(4, len(configs))):
        if time.time() - t0 > TIME_LIMIT - 40:
            break
        cfg = configs[start_idx]
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = cfg[i][0]
            x0[3*i+1] = cfg[i][1]
            x0[3*i+2] = cfg[i][2]
        
        try:
            result = differential_evolution(
                de_obj, bounds, x0=x0, maxiter=20, popsize=8,
                tol=1e-4, seed=100 + start_idx, polish=False
            )
            x_opt = result.x
            cx_opt = x_opt[0::3]
            cy_opt = x_opt[1::3]
            ang_opt = x_opt[2::3]
            
            cfg_opt = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
            eval_result = _evaluate_config(cfg_opt, t0)
            if eval_result is not None:
                side, ocx, ocy, oa, _, _, _ = eval_result
                if side < best_side:
                    best_side = side
                    best_cfg = cfg_opt
                    best_center = (ocx, ocy)
                    best_angle = oa
        except Exception:
            pass
    
    # Phase 3: Polish best config
    if best_cfg is not None and time.time() - t0 < TIME_LIMIT - 20:
        polish_result = _polish(best_cfg, t0, max_iter=300)
        if polish_result is not None:
            side, ocx, ocy, oa, cx_opt, cy_opt, ang_opt = polish_result
            if side < best_side:
                best_side = side
                best_cfg = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
                best_center = (ocx, ocy)
                best_angle = oa
    
    # Final fallback
    if best_cfg is None:
        best_cfg = configs[0]
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        result = _evaluate_config(best_cfg, t0)
        if result is not None:
            best_side, best_center[0], best_center[1], best_angle = result[0], result[1], result[2], result[3]
    
    return best_cfg, best_center, best_side, best_angle
