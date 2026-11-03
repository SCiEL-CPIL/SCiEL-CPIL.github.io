import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 142.0  # Safety margin below 150s

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

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check using distance-based pre-filter + SAT fallback.
    Returns True if all hexagons are pairwise disjoint.
    """
    n = len(cx_arr)
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
            
            if dist_sq < 4.0 - 1e-9:
                return False
            
            found_sep = False
            for axis in ai:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    found_sep = True
                    break
            if found_sep:
                continue
            
            for axis in aj:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    found_sep = True
                    break
            if not found_sep:
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

def _sdf_hex_point(px, py, angle_deg, side=1.0):
    """
    Signed distance from point (px,py) to a unit regular hexagon centered at origin
    with rotation angle_deg. Negative if inside, positive if outside.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    lx = ca * px + sa * py
    ly = -sa * px + ca * py
    apothem = side * math.sqrt(3) / 2.0
    
    # Check 6 half-planes defining the hexagon
    # Outward normals at 30, 90, 150, 210, 270, 330 degrees
    max_proj = 0.0
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx, ny = math.cos(theta), math.sin(theta)
        proj = nx * lx + ny * ly
        if proj > max_proj:
            max_proj = proj
    
    if max_proj <= apothem + 1e-12:
        # Inside: negative distance to nearest edge
        return -(apothem - max_proj)
    else:
        # Outside: need distance to polygon boundary
        # Compute distance to each edge segment
        min_d = float('inf')
        for k in range(6):
            t1 = math.radians(60 * k)
            t2 = math.radians(60 * (k + 1))
            v1x = side * math.cos(t1)
            v1y = side * math.sin(t1)
            v2x = side * math.cos(t2)
            v2y = side * math.sin(t2)
            
            ex = v2x - v1x
            ey = v2y - v1y
            len2 = ex * ex + ey * ey
            if len2 < 1e-12:
                continue
            
            t = ((lx - v1x) * ex + (ly - v1y) * ey) / len2
            t = max(0.0, min(1.0, t))
            qx = v1x + t * ex
            qy = v1y + t * ey
            d = math.hypot(lx - qx, ly - qy)
            if d < min_d:
                min_d = d
        
        return min_d

def _sdf_hex_hex(cx1, cy1, a1, cx2, cy2, a2, side=1.0):
    """
    Approximate SDF between two hexagons.
    Returns negative if overlapping, positive if separated.
    Uses vertex sampling of both hexagons.
    """
    verts1 = _hex_vertices(cx1, cy1, a1, side)
    verts2 = _hex_vertices(cx2, cy2, a2, side)
    
    min_d = float('inf')
    
    # For each vertex of hex1, compute SDF to hex2
    for (vx, vy) in verts1:
        d = _sdf_hex_point(vx - cx2, vy - cy2, a2, side)
        if d < min_d:
            min_d = d
    
    # For each vertex of hex2, compute SDF to hex1
    for (vx, vy) in verts2:
        d = _sdf_hex_point(vx - cx1, vy - cy1, a1, side)
        if d < min_d:
            min_d = d
    
    # Also check midpoints of edges for better accuracy
    for i in range(6):
        # Midpoint of edge i of hex1
        mx1 = (verts1[i][0] + verts1[(i+1)%6][0]) / 2.0
        my1 = (verts1[i][1] + verts1[(i+1)%6][1]) / 2.0
        d = _sdf_hex_point(mx1 - cx2, my1 - cy2, a2, side)
        if d < min_d:
            min_d = d
        
        # Midpoint of edge i of hex2
        mx2 = (verts2[i][0] + verts2[(i+1)%6][0]) / 2.0
        my2 = (verts2[i][1] + verts2[(i+1)%6][1]) / 2.0
        d = _sdf_hex_point(mx2 - cx1, my2 - cy1, a1, side)
        if d < min_d:
            min_d = d
    
    return min_d

def _smooth_collision_penalty(cfg, sigma=0.05):
    """
    Smooth penalty for collisions using exp-based softening.
    cfg: list of [x, y, angle] for N hexagons.
    Returns total penalty (0 if no collisions).
    """
    m = len(cfg)
    total_penalty = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            d = _sdf_hex_hex(cfg[i][0], cfg[i][1], cfg[i][2],
                           cfg[j][0], cfg[j][1], cfg[j][2])
            if d < 3.0 * sigma:
                # Smooth penalty: grows rapidly when d < 0
                penalty = math.exp(-d / sigma) - 1.0
                total_penalty += penalty
    return total_penalty

def _combined_objective(v, sigma=0.05):
    """
    Combined smooth objective: outer side length + collision penalty.
    v: flat array of [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    """
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    # Outer side with center at origin, angle 0
    side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
    
    # Collision penalty
    penalty = _smooth_collision_penalty(cfg, sigma)
    
    return side + 10.0 * penalty

def _make_flower_initial():
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

def _make_lattice_initial():
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

def _make_ring_initial():
    pts = [[0.0, 0.0, 0.0]]
    r = 2 * math.sqrt(3)
    for i in range(11):
        angle = math.radians(360 * i / 11 + 15)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        pts.append([x, y, 0.0])
    return pts

def _make_random_initial():
    pts = []
    for _ in range(N):
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        a = random.uniform(0, 60)
        pts.append([x, y, a])
    return pts

def _lbfgs_optimize(cfg, t0, time_budget=30.0):
    """
    Run L-BFGS-B on the smooth combined objective starting from cfg.
    Returns optimized cfg and final objective value.
    """
    v0 = np.array([val for p in cfg for val in p])
    
    result = minimize(
        _combined_objective,
        v0,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 5000}
    )
    
    v_opt = result.x
    cfg_opt = []
    for i in range(N):
        cfg_opt.append([v_opt[3*i], v_opt[3*i+1], v_opt[3*i+2]])
    
    return cfg_opt, result.fun

def _sa_refine(cfg, t0, time_budget=40.0):
    """
    Simulated annealing refinement using exact SAT validity check.
    Returns best valid configuration found.
    """
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    # Check initial validity
    if not _is_valid_fast(cx_arr, cy_arr, angle_arr):
        # Try to fix by scaling up
        scale = 1.1
        for _ in range(20):
            cx_arr *= scale
            cy_arr *= scale
            if _is_valid_fast(cx_arr, cy_arr, angle_arr):
                break
            scale += 0.05
    
    cur_score = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
    best_score = cur_score
    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
    
    T = 0.15
    T_end = 0.002
    alpha = 0.997
    
    while time.time() - t0 < TIME_LIMIT - 20:
        r = random.random()
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        idx2 = -1
        old_x2, old_y2, old_a2 = 0, 0, 0
        
        if r < 0.7:
            cx_arr[idx] += random.gauss(0, 0.10)
            cy_arr[idx] += random.gauss(0, 0.10)
            angle_arr[idx] += random.gauss(0, 2.0)
        elif r < 0.85:
            angle_arr[idx] += random.gauss(0, 5.0)
        else:
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, 0.07)
            cy_arr[idx] += random.gauss(0, 0.07)
            cx_arr[idx2] += random.gauss(0, 0.07)
            cy_arr[idx2] += random.gauss(0, 0.07)
            angle_arr[idx] += random.gauss(0, 1.5)
            angle_arr[idx2] += random.gauss(0, 1.5)
        
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if idx2 != -1:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        else:
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            if idx2 != -1:
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        
        T = max(T_end, T * alpha)
    
    return best_cfg, best_score

def _optimize_outer_hexagon(cfg, t0, time_budget=10.0):
    """
    Optimize the outer hexagon center and angle for a given inner configuration.
    Returns (best_side, best_center, best_angle).
    """
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    verts_list = [_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]) for i in range(N)]
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for outer_angle in range(0, 30, 2):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        # Find optimal center by iterative refinement
        cx_c, cy_c = 0.0, 0.0
        for _iter in range(15):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_list[i]:
                    for (nx, ny) in normals:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            for dx in [-0.2, -0.1, 0, 0.1, 0.2]:
                for dy in [-0.2, -0.1, 0, 0.1, 0.2]:
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_list[i]:
                            for (nx, ny) in normals:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                break
            cx_c, cy_c = best_cx, best_cy
        
        if max_proj < best_side:
            best_side = max_proj
            best_center = (cx_c, cy_c)
            best_angle = float(outer_angle)
    
    return best_side, best_center, best_angle

def construct():
    t0 = time.time()
    random.seed(123)
    
    initializers = [
        _make_flower_initial,
        _make_lattice_initial,
        _make_ring_initial,
        _make_random_initial,
        _make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Phase 0: Try initializers
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 80:
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
                best_cfg = [row[:] for row in cfg]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
    if best_cfg is None:
        cfg = _make_lattice_initial()
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
    
    # Phase 1: L-BFGS-B smooth optimization from multiple starting points
    lbfgs_starts = [
        best_cfg,
        _make_flower_initial(),
        _make_lattice_initial(),
        _make_ring_initial(),
    ]
    
    for start_cfg in lbfgs_starts:
        if time.time() - t0 > TIME_LIMIT - 50:
            break
        
        try:
            opt_cfg, obj_val = _lbfgs_optimize(start_cfg, t0, time_budget=20.0)
        except Exception:
            continue
        
        # Check if the optimized config is valid
        cx_arr = np.array([p[0] for p in opt_cfg])
        cy_arr = np.array([p[1] for p in opt_cfg])
        angle_arr = np.array([p[2] for p in opt_cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = [row[:] for row in opt_cfg]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
    # Phase 2: SA refinement from best L-BFGS result
    if time.time() - t0 < TIME_LIMIT - 30:
        sa_cfg, sa_side = _sa_refine(best_cfg, t0, time_budget=35.0)
        if sa_side < best_side:
            best_side = sa_side
            best_cfg = sa_cfg
            best_center = (0.0, 0.0)
            best_angle = 0.0
    
    # Phase 3: Optimize outer hexagon center and angle
    if time.time() - t0 < TIME_LIMIT - 5:
        final_side, final_center, final_angle = _optimize_outer_hexagon(best_cfg, t0, time_budget=10.0)
        if final_side < best_side:
            best_side = final_side
            best_center = final_center
            best_angle = final_angle
    
    # Final verification
    cx_final = np.array([p[0] for p in best_cfg])
    cy_final = np.array([p[1] for p in best_cfg])
    angle_final = np.array([p[2] for p in best_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        # Fallback: scale up until valid
        cfg = best_cfg[:]
        scale = 1.1
        for _ in range(30):
            for p in cfg:
                p[0] *= scale
                p[1] *= scale
            cx_arr = np.array([p[0] for p in cfg])
            cy_arr = np.array([p[1] for p in cfg])
            angle_arr = np.array([p[2] for p in cfg])
            if _is_valid_fast(cx_arr, cy_arr, angle_arr):
                break
            scale += 0.05
        best_cfg = cfg
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0
    else:
        # Recompute side with final center/angle
        best_side = _outer_hex_side(cx_final, cy_final, angle_final, best_center, best_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
