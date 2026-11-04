import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 146.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS  # ~1.732

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

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
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
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            
            # Quick distance check to skip far apart
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
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
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # 1. Initialize
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 100:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _quick_score(cx_arr, cy_arr, angle_arr)
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
        best_side = _quick_score(cx_arr, cy_arr, angle_arr)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # --- Phase 1: Simulated Annealing ---
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.3
    T_end = 0.001
    alpha = 0.995
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    def _quick_score_sa(cx_arr, cy_arr, angle_arr):
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

    while time.time() - t0 < TIME_LIMIT - 35:
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
            new_score = _quick_score_sa(cx_arr, cy_arr, angle_arr)
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
    
    # --- Phase 2: Joint Optimization with SDF-based collision penalty ---
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    def _joint_objective_and_grad(params):
        n_params = 3 * N + 2
        if len(params) != n_params:
            return 0, np.zeros(n_params)
        
        xs = params[0:N]
        ys = params[N:2*N]
        angles = params[2*N:3*N]
        cx_out = params[3*N]
        cy_out = params[3*N + 1]
        
        a_out = 0.0
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a_out + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        max_need = 0.0
        for i in range(N):
            verts = _hex_vertices(xs[i], ys[i], angles[i])
            for (vx, vy) in verts:
                for (nx, ny) in normals_out:
                    val = (nx * (vx - cx_out) + ny * (vy - cy_out)) / apothem_unit
                    if val > max_need:
                        max_need = val
        
        penalty = 0.0
        grad = np.zeros(n_params)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq > (2 * math.sqrt(3)) ** 2:
                    continue
                
                min_dist = 2.0
                if dist_sq < min_dist * min_dist:
                    dist = math.sqrt(dist_sq)
                    overlap = min_dist - dist
                    if overlap > 0:
                        penalty += overlap * overlap
                        if dist > 1e-6:
                            gx = dx / dist
                            gy = dy / dist
                            grad[i] += 2 * overlap * gx
                            grad[N + i] += 2 * overlap * gy
                            grad[j] -= 2 * overlap * gx
                            grad[N + j] -= 2 * overlap * gy
        
        penalty_weight = 10.0
        obj = max_need + penalty_weight * penalty
        
        return obj, grad
    
    from scipy.optimize import minimize
    
    init_params = np.zeros(3 * N + 2)
    for i in range(N):
        init_params[i] = cx_best[i]
        init_params[N + i] = cy_best[i]
        init_params[2 * N + i] = angle_best[i]
    init_params[3 * N] = 0.0
    init_params[3 * N + 1] = 0.0
    
    if time.time() - t0 < TIME_LIMIT - 10:
        try:
            result = minimize(
                _joint_objective_and_grad,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            if result.success or result.fun < best_side:
                opt_params = result.x
                xs_opt = opt_params[0:N]
                ys_opt = opt_params[N:2*N]
                angles_opt = opt_params[2*N:3*N]
                cx_out_opt = opt_params[3*N]
                cy_out_opt = opt_params[3*N + 1]
                
                if _is_valid_fast(xs_opt, ys_opt, angles_opt):
                    side_opt = _outer_hex_side(xs_opt, ys_opt, angles_opt, (cx_out_opt, cy_out_opt), 0.0)
                    if side_opt < best_final_side:
                        best_final_side = side_opt
                        best_final_cfg = [[xs_opt[i], ys_opt[i], angles_opt[i]] for i in range(N)]
                        best_final_center = (cx_out_opt, cy_out_opt)
                        best_final_angle = 0.0
        except Exception:
            pass
    
    # --- Phase 3: Refine outer hexagon center and angle ---
    cx_best2 = np.array([p[0] for p in best_final_cfg])
    cy_best2 = np.array([p[1] for p in best_final_cfg])
    angle_best2 = np.array([p[2] for p in best_final_cfg])
    
    verts_best = [_hex_vertices(cx_best2[i], cy_best2[i], angle_best2[i]) for i in range(N)]
    
    for outer_angle in range(0, 30, 2):
        if time.time() - t0 > TIME_LIMIT - 3:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = best_final_center[0], best_final_center[1]
        for _iter in range(20):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
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
                        for (vx, vy) in verts_best[i]:
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
        
        if max_proj < best_final_side:
            best_final_side = max_proj
            best_final_center = (cx_c, cy_c)
            best_final_angle = float(outer_angle)
    
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

BET: {mechanism: "Vectorized pre-check in collision detection reduces overhead, allowing more simulated annealing iterations within the time limit, leading to better packing configurations.", q: 0.7, delta: [0.05, 0.3]}
