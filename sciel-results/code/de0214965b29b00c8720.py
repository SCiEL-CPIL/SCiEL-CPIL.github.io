"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, linprog

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Return True if they overlap."""
    # Quick reject
    cxA = sum(v[0] for v in vertsA) / len(vertsA)
    cyA = sum(v[1] for v in vertsA) / len(vertsA)
    cxB = sum(v[0] for v in vertsB) / len(vertsB)
    cyB = sum(v[1] for v in vertsB) / len(vertsB)
    dist = math.hypot(cxA - cxB, cyA - cyB)
    if dist > 2.0 + 1e-9:
        return False
    
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + tol or maxB <= minA + tol:
            return False
    return True

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j], tol):
                return False
    return True

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _chebyshev_center_lp(normals, all_verts):
    """
    Find the Chebyshev center (minimize max projection) using scipy LP.
    Minimize t subject to n_k · (v - c) <= t * apothem for all k, v.
    Variables: [cx, cy, t]
    """
    apothem_unit = math.sqrt(3) / 2.0
    n_verts = len(all_verts)
    n_norms = len(normals)
    
    # Minimize t (index 2)
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # A_ub @ x <= b_ub
    # n_kx*vx + n_ky*vy - n_kx*cx - n_ky*cy <= t * apothem
    # -n_kx*cx - n_ky*cy - apothem*t <= -n_kx*vx - n_ky*vy
    A_rows = []
    b_rows = []
    for (nx, ny) in normals:
        for (vx, vy) in all_verts:
            A_rows.append([-nx, -ny, -apothem_unit])
            b_rows.append(-(nx * vx + ny * vy))
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return float(res.x[0]), float(res.x[1]), float(res.x[2] * apothem_unit)
    except Exception:
        pass
    return None, None, None

def _optimize_composite_body(cfg):
    """
    Optimize outer hexagon center and rotation for a fixed internal configuration.
    Uses Golden Section Search on rotation angle, and LP for center.
    """
    all_verts = []
    for (px, py, pangle) in cfg:
        verts = _hex_vertices(px, py, pangle)
        all_verts.extend(verts)
    
    # Centroid as initial guess
    cx_body = sum(v[0] for v in all_verts) / len(all_verts)
    cy_body = sum(v[1] for v in all_verts) / len(all_verts)
    
    def side_for_angle(angle_deg):
        a = math.radians(angle_deg)
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        cx, cy, side = _chebyshev_center_lp(normals, all_verts)
        if side is None:
            # Fallback to heuristic if LP fails
            cx, cy = cx_body, cy_body
            apothem_unit = math.sqrt(3) / 2.0
            max_need = 0.0
            for (vx, vy) in all_verts:
                for (nx, ny) in normals:
                    val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                    if val > max_need:
                        max_need = val
            side = max_need
        return side, (cx, cy)
    
    # Golden section search on [0, 60)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a = 0.0
    b = 60.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    fc, center_c = side_for_angle(c)
    fd, center_d = side_for_angle(d)
    
    best_side = min(fc, fd)
    best_center = center_c if fc < fd else center_d
    best_angle = c if fc < fd else d
    
    for _ in range(100):
        if b - a < 1e-7:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc, center_c = side_for_angle(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd, center_d = side_for_angle(d)
        
        if fc < best_side:
            best_side = fc
            best_center = center_c
            best_angle = c
        if fd < best_side:
            best_side = fd
            best_center = center_d
            best_angle = d
            
    return best_center, best_angle, best_side

def _smooth_outer_side(cfg, center, angle_deg, beta=10.0):
    """Smooth approximation of outer hex side."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    vals = []
    for (px, py, pangle) in cfg:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                vals.append(val)
    
    vals = np.array(vals)
    max_val = np.max(vals)
    shifted = vals - max_val
    lse = max_val + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
    return lse

def _smooth_collision_penalty(cfg, beta=10.0):
    """Smooth collision penalty."""
    m = len(cfg)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in cfg]
    penalty = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            axes = _axes_from_polygon(polys[i]) + _axes_from_polygon(polys[j])
            min_sep = float('inf')
            for axis in axes:
                minA, maxA = _project_polygon(polys[i], axis)
                minB, maxB = _project_polygon(polys[j], axis)
                sep = min(maxA, maxB) - max(minA, minB)
                if sep < min_sep:
                    min_sep = sep
            if min_sep < 0:
                penalty += min_sep ** 2
    return penalty

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def full_score(cfg, center, angle):
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    def make_hex_lattice():
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts

    def make_ring():
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts

    def make_flower():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    def make_dense_cluster():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = math.sqrt(3) * math.cos(theta)
            y = math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = 2 * math.sqrt(3) * math.cos(theta)
            y = 2 * math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts

    inits = [make_hex_lattice(), make_ring(), make_flower(), make_dense_cluster()]
    
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
        
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    # Simulated Annealing
    T = 0.8
    T_end = 0.003
    alpha = 0.9997
    
    sa_time_limit = TIME_LIMIT - 30.0
    
    # Precompute lattice positions for snap moves
    lattice_positions = []
    row_height = math.sqrt(3)
    col_width = 2.0
    for r in range(5):
        for c in range(6):
            x = c * col_width + (col_width / 2 if r % 2 else 0.0)
            y = r * row_height
            lattice_positions.append((x - 3.0, y - 2.0 * row_height))
            
    while time.time() - t0 < sa_time_limit:
        r = random.random()
        
        if r < 0.45:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.2 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.60:
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
        elif r < 0.70:
            old_cx, old_cy = center[0], center[1]
            step = 0.1 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.80:
            old_angle = angle
            angle += random.gauss(0, 2.0 + 2.0 * T)
        elif r < 0.90:
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        else:
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            best_lp = None
            best_dist = float('inf')
            for lx, ly in lattice_positions:
                d = (cfg[idx][0] - lx)**2 + (cfg[idx][1] - ly)**2
                if d < best_dist:
                    best_dist = d
                    best_lp = (lx, ly)
            if best_lp is not None:
                cfg[idx][0] = cfg[idx][0] + 0.8 * (best_lp[0] - cfg[idx][0])
                cfg[idx][1] = cfg[idx][1] + 0.8 * (best_lp[1] - cfg[idx][1])
        
        new_score = full_score(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                if r < 0.45:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.60:
                    cfg[idx][2] = old_ang
                elif r < 0.70:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.80:
                    angle = old_angle
                elif r < 0.90:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                else:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
        else:
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.60:
                cfg[idx][2] = old_ang
            elif r < 0.70:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.80:
                angle = old_angle
            elif r < 0.90:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
            else:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
                
        T = max(T_end, T * alpha)
        
    # Composite body optimization
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            cb_center, cb_angle, cb_side = _optimize_composite_body(best_cfg)
            if cb_side < best_score:
                if _is_valid(best_cfg):
                    best_score = cb_side
                    best_center = cb_center
                    best_angle = cb_angle
        except Exception:
            pass

    # Nelder-Mead polish on internal variables
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 15.0:
        try:
            def full_nelder_obj(params):
                cfg_local = []
                for i in range(N):
                    cfg_local.append([params[3*i], params[3*i+1], params[3*i+2]])
                cx, cy = params[3*N], params[3*N+1]
                oa = params[3*N+2]
                
                if not _is_valid(cfg_local):
                    return 1000.0
                
                return _outer_hex_side(cfg_local, (cx, cy), oa)
            
            x0_full = []
            for p in best_cfg:
                x0_full.extend([p[0], p[1], p[2]])
            x0_full.extend([best_center[0], best_center[1], best_angle])
            x0_full = np.array(x0_full, dtype=float)
            
            res_full = minimize(
                full_nelder_obj, x0_full, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_full.fun < best_score and res_full.fun < 100.0:
                cfg_new = []
                for i in range(N):
                    cfg_new.append([float(res_full.x[3*i]), float(res_full.x[3*i+1]), float(res_full.x[3*i+2])])
                c_new = [float(res_full.x[3*N]), float(res_full.x[3*N+1])]
                a_new = float(res_full.x[3*N+2]) % 60.0
                
                if _is_valid(cfg_new):
                    s_new = _outer_hex_side(cfg_new, tuple(c_new), a_new)
                    if s_new < best_score:
                        best_score = s_new
                        best_cfg = cfg_new
                        best_center = c_new
                        best_angle = a_new
        except Exception:
            pass
            
    # Final composite optimization
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            cb_center2, cb_angle2, cb_side2 = _optimize_composite_body(best_cfg)
            if cb_side2 < best_score:
                if _is_valid(best_cfg):
                    best_score = cb_side2
                    best_center = cb_center2
                    best_angle = cb_angle2
        except Exception:
            pass

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
