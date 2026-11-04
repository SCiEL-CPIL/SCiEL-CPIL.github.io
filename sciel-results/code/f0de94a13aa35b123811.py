import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS  # sqrt(3) ~ 1.732

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

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check using distance-based pre-filter + SAT fallback.
    cx_arr, cy_arr, angle_arr: numpy arrays of shape (N,)
    Returns True if all hexagons are pairwise disjoint.
    """
    n = len(cx_arr)
    
    # Fast pre-check: Circle packing (vectorized)
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9
    
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False

    # Exact check: SAT (optimized)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            aj = axes_list[j]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
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
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
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

def _quick_score(cx_arr, cy_arr, angle_arr):
    """Approximate score assuming outer hex is centered at (0,0) and aligned 0."""
    apothem_unit = math.sqrt(3) / 2.0
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    max_need = 0.0
    for i in range(N):
        a = math.radians(angle_arr[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * vx + ny * vy) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _sdf_hexagon_point(px, py, cx, cy, angle_deg):
    """
    Signed distance from point (px,py) to a unit regular hexagon centered at (cx,cy)
    with rotation angle_deg. Negative if inside, positive if outside.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx = px - cx
    dy = py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    apothem = math.sqrt(3) / 2.0
    
    max_dist = 0.0
    for k in range(6):
        na = math.radians(60 * k + 30)
        nx, ny = math.cos(na), math.sin(na)
        dist = nx * lx + ny * ly - apothem
        if dist > max_dist:
            max_dist = dist
    
    if max_dist < 0:
        min_dist_to_edge = float('inf')
        for k in range(6):
            na = math.radians(60 * k + 30)
            nx, ny = math.cos(na), math.sin(na)
            dist = apothem - (nx * lx + ny * ly)
            if dist > 0 and dist < min_dist_to_edge:
                min_dist_to_edge = dist
        return -min_dist_to_edge
    else:
        return max_dist

def _sdf_hexagon_hexagon(cx1, cy1, a1, cx2, cy2, a2):
    """
    Approximate signed distance between two unit hexagons.
    Uses the minimum SDF from vertices of one hex to the other hexagon.
    """
    verts1 = _hex_vertices(cx1, cy1, a1)
    verts2 = _hex_vertices(cx2, cy2, a2)
    
    min_sdf = float('inf')
    for (vx, vy) in verts1:
        d = _sdf_hexagon_point(vx, vy, cx2, cy2, a2)
        if d < min_sdf:
            min_sdf = d
    for (vx, vy) in verts2:
        d = _sdf_hexagon_point(vx, vy, cx1, cy1, a1)
        if d < min_sdf:
            min_sdf = d
    
    d_center = math.hypot(cx1 - cx2, cy1 - cy2)
    if d_center > 2 * math.sqrt(3):
        return d_center - 2 * math.sqrt(3)
    
    return min_sdf

def _nm_objective(params, outer_center, outer_angle):
    """
    Objective for Nelder-Mead: minimize outer hex side length + collision penalty.
    params: flat array of [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    """
    n = len(params) // 3
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    
    a_outer = math.radians(outer_angle)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a_outer + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx_c, cy_c = outer_center
    max_need = 0.0
    for i in range(n):
        verts = _hex_vertices(cx[i], cy[i], ang[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                if val > max_need:
                    max_need = val
    
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = _sdf_hexagon_hexagon(cx[i], cy[i], ang[i], cx[j], cy[j], ang[j])
            if d < 0:
                penalty += d * d
    
    objective = max_need + 100.0 * penalty
    return objective

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

    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
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
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # --- Phase 1: Simulated Annealing ---
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.2
    T_end = 0.003
    alpha = 0.996
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    def _quick_score_local(cx_arr, cy_arr, angle_arr):
        max_need = 0.0
        for i in range(N):
            a = math.radians(angle_arr[i])
            ca, sa = math.cos(a), math.sin(a)
            for k in range(6):
                theta = math.radians(60 * k)
                vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
                vy = cx_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
                for (nx, ny) in normals:
                    val = (nx * vx + ny * vy) / apothem_unit
                    if val > max_need:
                        max_need = val
        return max_need

    while time.time() - t0 < TIME_LIMIT - 30:
        r = random.random()
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        idx2 = -1
        old_x2, old_y2, old_a2 = 0, 0, 0
        
        if r < 0.7:
            cx_arr[idx] += random.gauss(0, 0.12)
            cy_arr[idx] += random.gauss(0, 0.12)
            angle_arr[idx] += random.gauss(0, 2.5)
        elif r < 0.85:
            angle_arr[idx] += random.gauss(0, 6.0)
        else:
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, 0.08)
            cy_arr[idx] += random.gauss(0, 0.08)
            cx_arr[idx2] += random.gauss(0, 0.08)
            cy_arr[idx2] += random.gauss(0, 0.08)
            angle_arr[idx] += random.gauss(0, 1.5)
            angle_arr[idx2] += random.gauss(0, 1.5)
        
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score_local(cx_arr, cy_arr, angle_arr)
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if idx2 != -1:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        else:
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            if idx2 != -1:
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        
        T = max(T_end, T * alpha)
    
    # --- Phase 2: Nelder-Mead optimization ---
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = cx_best[i]
        x0[3*i+1] = cy_best[i]
        x0[3*i+2] = angle_best[i]
    
    for outer_angle_deg in [0, 5, 10, 15, 20, 25]:
        if time.time() - t0 > TIME_LIMIT - 20:
            break
        
        outer_center = (0.0, 0.0)
        
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                _nm_objective,
                x0,
                args=(outer_center, float(outer_angle_deg)),
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-8, 'xtol': 1e-6}
            )
            
            if result.fun < best_final_side:
                opt_params = result.x
                opt_cfg = []
                for i in range(N):
                    opt_cfg.append([opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]])
                
                opt_cx = np.array([p[0] for p in opt_cfg])
                opt_cy = np.array([p[1] for p in opt_cfg])
                opt_ang = np.array([p[2] for p in opt_cfg])
                
                if _is_valid_fast(opt_cx, opt_cy, opt_ang):
                    side = _outer_hex_side(opt_cx, opt_cy, opt_ang, outer_center, float(outer_angle_deg))
                    if side < best_final_side:
                        best_final_side = side
                        best_final_cfg = opt_cfg[:]
                        best_final_center = outer_center
                        best_final_angle = float(outer_angle_deg)
                        x0 = opt_params[:]
        except Exception:
            pass
    
    # Phase 3: Refine outer hexagon center for the best configuration found
    if best_final_cfg is not None:
        cx_ref = np.array([p[0] for p in best_final_cfg])
        cy_ref = np.array([p[1] for p in best_final_cfg])
        ang_ref = np.array([p[2] for p in best_final_cfg])
        
        verts_ref = [_hex_vertices(cx_ref[i], cy_ref[i], ang_ref[i]) for i in range(N)]
        
        outer_a = math.radians(best_final_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals_ref = []
        for k in range(6):
            na = outer_a + math.radians(90 + 60 * k)
            normals_ref.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = best_final_center
        
        for _iter in range(30):
            if time.time() - t0 > TIME_LIMIT - 5:
                break
            
            best_cx, best_cy, best_val = cx_c, cy_c, float('inf')
            
            for dx in np.linspace(-0.3, 0.3, 7):
                for dy in np.linspace(-0.3, 0.3, 7):
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_ref[i]:
                            for (nx, ny) in normals_ref:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                break
            cx_c, cy_c = best_cx, best_cy
        
        if best_val < best_final_side:
            best_final_side = best_val
            best_final_center = (cx_c, cy_c)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        cfg = make_lattice_initial()
        scale = 1.3
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
