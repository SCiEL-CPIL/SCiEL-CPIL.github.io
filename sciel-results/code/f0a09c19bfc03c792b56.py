import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 115.0  # Safety margin below 150s (evaluator kills at 150s, need buffer)
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS  # sqrt(3) ~ 1.732

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

def _sdf_hexagon_point(px, py, cx, cy, angle_deg):
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
    d_center = math.hypot(cx1 - cx2, cy1 - cy2)
    if d_center > 2 * math.sqrt(3):
        return d_center - 2 * math.sqrt(3)
    
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
    return min_sdf

def _nm_objective(params, outer_angle_deg):
    cx_arr = params[0::3]
    cy_arr = params[1::3]
    ang_arr = params[2::3]
    a_outer = math.radians(outer_angle_deg)
    apothem = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a_outer + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    max_need = 0.0
    for i in range(N):
        a = math.radians(ang_arr[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * vx + ny * vy) / apothem
                if val > max_need:
                    max_need = val
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _sdf_hexagon_hexagon(cx_arr[i], cy_arr[i], ang_arr[i],
                                     cx_arr[j], cy_arr[j], ang_arr[j])
            if d < 0:
                penalty += d * d
    return max_need + 50.0 * penalty

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

    # 2. Differential Evolution (Global Search) - Structural change from SA
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    pop_size = 12
    population = []
    population.append([cx_arr.copy(), cy_arr.copy(), angle_arr.copy(), best_side])
    
    for _ in range(pop_size - 1):
        p = best_cfg[:]
        for i in range(N):
            p[i][0] += random.gauss(0, 0.8)
            p[i][1] += random.gauss(0, 0.8)
            p[i][2] += random.gauss(0, 10.0)
        
        cx_p = np.array([q[0] for q in p])
        cy_p = np.array([q[1] for q in p])
        ang_p = np.array([q[2] for q in p])
        
        if _is_valid_fast(cx_p, cy_p, ang_p):
            score = _quick_score(cx_p, cy_p, ang_p)
            population.append([cx_p, cy_p, ang_p, score])
        else:
            population.append([cx_p, cy_p, ang_p, float('inf')])
    
    # Sort population
    population.sort(key=lambda x: x[3])
    
    de_time_budget = 30.0
    de_start = time.time()
    
    while time.time() - de_start < de_time_budget:
        if time.time() - t0 > TIME_LIMIT - 40:
            break
        
        # DE step
        new_pop = []
        for i in range(pop_size):
            # Mutation: DE/rand/1
            a, b, c = random.sample(range(pop_size), 3)
            if a == i or b == i or c == i:
                continue
            
            F = 0.8
            new_cx = population[a][0].copy()
            new_cy = population[a][1].copy()
            new_ang = population[a][2].copy()
            
            for k in range(N):
                new_cx[k] = population[a][0][k] + F * (population[b][0][k] - population[c][0][k])
                new_cy[k] = population[a][1][k] + F * (population[b][1][k] - population[c][1][k])
                new_ang[k] = population[a][2][k] + F * (population[b][2][k] - population[c][2][k])
            
            # Crossover
            CR = 0.5
            trial_cx = new_cx.copy()
            trial_cy = new_cy.copy()
            trial_ang = new_ang.copy()
            
            for k in range(3 * N):
                if random.random() < CR:
                    idx = k // 3
                    if k % 3 == 0:
                        trial_cx[idx] = new_cx[idx]
                    elif k % 3 == 1:
                        trial_cy[idx] = new_cy[idx]
                    else:
                        trial_ang[idx] = new_ang[idx]
                # else keep from current pop[i]
            
            # Evaluate
            if _is_valid_fast(trial_cx, trial_cy, trial_ang):
                score = _quick_score(trial_cx, trial_cy, trial_ang)
            else:
                score = float('inf')
            
            # Selection
            if score <= population[i][3]:
                new_pop.append([trial_cx, trial_cy, trial_ang, score])
                if score < best_side:
                    best_side = score
                    best_cfg = [[trial_cx[k], trial_cy[k], trial_ang[k]] for k in range(N)]
            else:
                new_pop.append(population[i])
        
        population = new_pop
        population.sort(key=lambda x: x[3])
    
    # 3. Nelder-Mead polish over multiple outer hexagon angles
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = (0.0, 0.0)
    best_final_angle = 0.0

    for outer_angle_deg in [0, 10, 20, 30, 40, 50]:
        if time.time() - t0 > TIME_LIMIT - 15:
            break

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = best_final_cfg[i][0]
            x0[3*i+1] = best_final_cfg[i][1]
            x0[3*i+2] = best_final_cfg[i][2]

        try:
            result = minimize(
                _nm_objective,
                x0,
                args=(float(outer_angle_deg),),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7,
                         'adaptive': True}
            )
            opt_params = result.x
            opt_cfg = []
            for i in range(N):
                opt_cfg.append([opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]])
            opt_cx = np.array([p[0] for p in opt_cfg])
            opt_cy = np.array([p[1] for p in opt_cfg])
            opt_ang = np.array([p[2] for p in opt_cfg])

            if _is_valid_fast(opt_cx, opt_cy, opt_ang):
                side = _outer_hex_side(opt_cx, opt_cy, opt_ang, (0.0, 0.0), float(outer_angle_deg))
                if side < best_final_side:
                    best_final_side = side
                    best_final_cfg = opt_cfg[:]
                    best_final_center = (0.0, 0.0)
                    best_final_angle = float(outer_angle_deg)
        except Exception:
            pass
    
    # Phase 3: Refine outer hexagon center and angle for the best configuration
    if best_final_cfg is not None:
        cx_ref = np.array([p[0] for p in best_final_cfg])
        cy_ref = np.array([p[1] for p in best_final_cfg])
        ang_ref = np.array([p[2] for p in best_final_cfg])

        verts_ref = [_hex_vertices(cx_ref[i], cy_ref[i], ang_ref[i]) for i in range(N)]

        cx_c, cy_c = best_final_center
        best_val = best_final_side

        for ang_offset in range(-15, 16, 3):
            if time.time() - t0 > TIME_LIMIT - 5:
                break
            outer_a = math.radians(best_final_angle + ang_offset)
            apothem_unit = math.sqrt(3) / 2.0
            normals_ref = []
            for k in range(6):
                na = outer_a + math.radians(90 + 60 * k)
                normals_ref.append((math.cos(na), math.sin(na)))

            for dx in np.linspace(-0.5, 0.5, 11):
                for dy in np.linspace(-0.5, 0.5, 11):
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
                        best_final_side = val
                        best_final_center = (ncx, ncy)
                        best_final_angle = best_final_angle + ang_offset
    
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
