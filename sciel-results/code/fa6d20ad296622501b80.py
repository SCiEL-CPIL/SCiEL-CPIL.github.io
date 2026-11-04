import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 115.0  # Safety margin below 120s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS

def _hex_vertices(cx, cy, angle_deg, side=1.0):
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
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
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

def _is_valid_sat(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False
    
    verts_list = [_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]) for i in range(n)]
    axes_list = [_axes_from_polygon(verts_list[i]) for i in range(n)]
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            overlap = True
            for axis in ai:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    overlap = False
                    break
            if overlap:
                for axis in aj:
                    minA, maxA = _project_polygon(vi, axis)
                    minB, maxB = _project_polygon(vj, axis)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        overlap = False
                        break
            if overlap:
                return False
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _sdf_hex_point(px, py, angle_deg, side=1.0):
    """Signed distance from point to hexagon centered at origin with rotation."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    lx = ca * px + sa * py
    ly = -sa * px + ca * py
    apothem = side * math.sqrt(3) / 2.0
    
    # Check if inside using 6 half-planes
    inside = True
    max_proj = 0.0
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx, ny = math.cos(theta), math.sin(theta)
        proj = nx * lx + ny * ly
        if proj > apothem + 1e-12:
            inside = False
            break
        if proj > max_proj:
            max_proj = proj
    
    if inside:
        return -(apothem - max_proj)
    else:
        # Outside: min distance to edges
        min_d = float('inf')
        for k in range(6):
            t1 = math.radians(60 * k)
            t2 = math.radians(60 * (k + 1))
            v1 = (side * math.cos(t1), side * math.sin(t1))
            v2 = (side * math.cos(t2), side * math.sin(t2))
            dx, dy = v2[0] - v1[0], v2[1] - v1[1]
            len2 = dx * dx + dy * dy
            if len2 < 1e-12:
                continue
            t = max(0.0, min(1.0, ((lx - v1[0]) * dx + (ly - v1[1]) * dy) / len2))
            qx, qy = v1[0] + t * dx, v1[1] + t * dy
            d = math.hypot(lx - qx, ly - qy)
            if d < min_d:
                min_d = d
        return min_d

def _sdf_hex_hex(cx1, cy1, a1, cx2, cy2, a2, side=1.0):
    """Min SDF between two hexagons using vertex sampling."""
    verts1 = _hex_vertices(cx1, cy1, a1, side)
    verts2 = _hex_vertices(cx2, cy2, a2, side)
    min_d = float('inf')
    for (vx, vy) in verts1:
        d = _sdf_hex_point(vx - cx2, vy - cy2, a2, side)
        if d < min_d:
            min_d = d
    for (vx, vy) in verts2:
        d = _sdf_hex_point(vx - cx1, vy - cy1, a1, side)
        if d < min_d:
            min_d = d
    return min_d

def _smooth_objective(v):
    """Smooth objective: outer side + collision penalty."""
    n = 12
    cfg = [[v[3*i], v[3*i+1], v[3*i+2]] for i in range(n)]
    center = (v[3*n], v[3*n+1])
    angle = v[3*n+2]
    
    cx_arr = np.array([c[0] for c in cfg])
    cy_arr = np.array([c[1] for c in cfg])
    angle_arr = np.array([c[2] for c in cfg])
    
    s = _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle)
    
    # Smooth collision penalty
    sigma = 0.15
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = _sdf_hex_hex(cfg[i][0], cfg[i][1], cfg[i][2],
                           cfg[j][0], cfg[j][1], cfg[j][2])
            if d < 3.0 * sigma:
                penalty += math.exp(-d / sigma) - 1.0
    
    return s + 50.0 * penalty

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_flower_initial():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        r2 = r1 * 1.732
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
        return pts[:12]
    
    def make_lattice_initial():
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        return pts
    
    def make_ring_initial():
        pts = [[0.0, 0.0, 0.0]]
        r = 2 * math.sqrt(3)
        for i in range(11):
            angle = math.radians(360 * i / 11 + 15)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            pts.append([x, y, 0.0])
        return pts
    
    initializers = [make_flower_initial, make_lattice_initial, make_ring_initial]
    
    best_side = float('inf')
    best_cfg = None
    
    # Phase 1: Initialize with valid configs
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 90:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_sat(cx_arr, cy_arr, angle_arr):
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0,0), 0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
    
    if best_cfg is None:
        cfg = make_lattice_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0,0), 0)
    
    # Phase 2: L-BFGS-B with smooth SDF objective
    # Optimize inner positions + outer center + outer angle
    def flatten(cfg, center, angle):
        v = []
        for p in cfg:
            v.extend([p[0], p[1], p[2]])
        v.extend([center[0], center[1], angle])
        return np.array(v)
    
    def unflatten(v):
        cfg = [[v[3*i], v[3*i+1], v[3*i+2]] for i in range(N)]
        center = (v[3*N], v[3*N+1])
        angle = v[3*N+2]
        return cfg, center, angle
    
    v0 = flatten(best_cfg, (0.0, 0.0), 0.0)
    
    # Multiple restarts with L-BFGS-B
    lbfgs_deadline = time.time() + 50
    for restart in range(3):
        if time.time() > lbfgs_deadline:
            break
        
        if restart == 0:
            v_start = v0.copy()
        else:
            v_start = v0.copy()
            for i in range(3*N):
                v_start[i] += random.gauss(0, 0.3)
        
        result = minimize(_smooth_objective, v_start, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result.success or result.fun < _smooth_objective(v0):
            cfg_r, center_r, angle_r = unflatten(result.x)
            cx_r = np.array([c[0] for c in cfg_r])
            cy_r = np.array([c[1] for c in cfg_r])
            ang_r = np.array([c[2] for c in cfg_r])
            
            if _is_valid_sat(cx_r, cy_r, ang_r):
                side_r = _outer_hex_side(cx_r, cy_r, ang_r, center_r, angle_r)
                if side_r < best_side:
                    best_side = side_r
                    best_cfg = cfg_r
                    v0 = result.x.copy()
    
    # Phase 3: SA refinement with SAT validity
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    cur_score = best_side
    T = 0.15
    T_end = 0.001
    alpha = 0.999
    
    sa_deadline = time.time() + 25
    
    while time.time() < sa_deadline:
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        
        if random.random() < 0.8:
            cx_arr[idx] += random.gauss(0, 0.08)
            cy_arr[idx] += random.gauss(0, 0.08)
            angle_arr[idx] += random.gauss(0, 1.5)
        else:
            idx2 = random.randrange(N)
            if idx2 == idx:
                T = max(T_end, T * alpha)
                continue
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            
            cx_arr[idx] += random.gauss(0, 0.08)
            cy_arr[idx] += random.gauss(0, 0.08)
            angle_arr[idx] += random.gauss(0, 1.5)
            
            cx_arr[idx2] += random.gauss(0, 0.08)
            cy_arr[idx2] += random.gauss(0, 0.08)
            angle_arr[idx2] += random.gauss(0, 1.5)
            
            valid = _is_valid_sat(cx_arr, cy_arr, angle_arr)
            
            if valid:
                new_score = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0,0), 0)
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    cur_score = new_score
                    if new_score < best_side:
                        best_side = new_score
                        best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                else:
                    cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            T = max(T_end, T * alpha)
            continue
        
        valid = _is_valid_sat(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0,0), 0)
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
        else:
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
        
        T = max(T_end, T * alpha)
    
    # Phase 4: Final polish - optimize outer hex center and angle
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = (0.0, 0.0)
    best_final_angle = 0.0
    
    verts_best = [_hex_vertices(cx_best[i], cy_best[i], angle_best[i]) for i in range(N)]
    
    polish_deadline = time.time() + 12
    
    for outer_angle in range(0, 30, 3):
        if time.time() > polish_deadline:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = 0.0, 0.0
        
        for _iter in range(8):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals_out:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            for dx in [-0.15, -0.08, 0.0, 0.08, 0.15]:
                for dy in [-0.15, -0.08, 0.0, 0.08, 0.15]:
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_best[i]:
                            for (nx, ny) in normals_out:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                break
            cx_c, cy_c = best_cx, best_cy
        
        if max_proj < best_final_side:
            best_final_side = max_proj
            best_final_center = (cx_c, cy_c)
            best_final_angle = float(outer_angle)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_sat(cx_final, cy_final, angle_final):
        cfg = make_lattice_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_final_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
    else:
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, best_final_center, best_final_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_final_cfg]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
