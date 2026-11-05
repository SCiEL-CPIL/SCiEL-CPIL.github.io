import numpy as np
import time
import math
import random
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def outer_hex_side(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def optimize_outer_rotation(points, init_rot=0.0, n_samples=720):
    best_L = float('inf')
    best_rot = init_rot
    rots = np.linspace(0, 60, n_samples, endpoint=False)
    for rot in rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    local_rots = np.linspace(best_rot - 2.0, best_rot + 2.0, n_samples)
    for rot in local_rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    return best_L, best_rot

def sat_overlap(v1, v2):
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    return -min_overlap

def is_valid(params, tol=1e-6):
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < -tol:
                return False
    return True

def get_L(params):
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    return outer_hex_side(all_points, outer_rot)

def objective_combined(params, penalty_weight=100.0, beta=40.0):
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(xys_rots[i, :2] - xys_rots[j, :2])
            if d < 2.5:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    if penetrations:
        p_arr = np.array(penetrations)
        max_pen = np.max(p_arr)
        if max_pen < 1e-12:
            penalty = 0.0
        else:
            shifted = beta * (p_arr - max_pen)
            shifted = np.clip(shifted, -50, 50)
            penalty = max_pen + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    else:
        penalty = 0.0
    all_points = np.vstack(vertices_list)
    L = outer_hex_side(all_points, outer_rot)
    return L + penalty_weight * penalty

def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
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
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_ring(gap=0.0):
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
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_ring12(gap=0.0):
    pts = []
    R = 2.2 + gap
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        angle_deg = math.degrees(theta)
        pts.append([x, y, angle_deg])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_compact_cluster(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    dx = sqrt3 + gap
    dy = 1.5 + gap * 0.5
    for c in range(3):
        pts.append([c * dx, 0, 0.0])
    for c in range(4):
        pts.append([c * dx + dx/2, dy, 0.0])
    for c in range(3):
        pts.append([c * dx, 2*dy, 0.0])
    for c in range(2):
        pts.append([c * dx + dx/2, 3*dy, 0.0])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_diamond(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    dx = sqrt3 + gap
    dy = 1.5 + gap * 0.5
    for c in range(2):
        pts.append([c * dx + dx/2, 0, 0.0])
    for c in range(3):
        pts.append([c * dx, dy, 0.0])
    for c in range(4):
        pts.append([c * dx + dx/2, 2*dy, 0.0])
    for c in range(3):
        pts.append([c * dx, 3*dy, 0.0])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_trapezoid(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    dx = sqrt3 + gap
    dy = 1.5 + gap * 0.5
    # Row 0: 4 hexes
    for c in range(4):
        pts.append([c * dx, 0, 0.0])
    # Row 1: 4 hexes offset
    for c in range(4):
        pts.append([c * dx + dx/2, dy, 0.0])
    # Row 2: 4 hexes
    for c in range(4):
        pts.append([c * dx, 2*dy, 0.0])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_spiral(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    dx = sqrt3 + gap
    dy = 1.5 + gap * 0.5
    # Spiral-like pattern
    positions = [
        (0, 0), (1, 0), (2, 0),
        (2.5, 1), (1.5, 1), (0.5, 1),
        (0, 2), (1, 2), (2, 2),
        (2.5, 3), (1.5, 3), (0.5, 3)
    ]
    for x_grid, y_grid in positions:
        x = x_grid * dx
        y = y_grid * dy
        pts.append([x, y, 0.0])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    def time_left():
        return DEADLINE - (time.time() - start_time)

    best_L = float('inf')
    best_params = None

    configs = []
    for gap in [0.02, 0.05, 0.1]:
        configs.append(make_hex_grid(4, 3, gap=gap))
        configs.append(make_hex_grid(3, 4, gap=gap))
        configs.append(make_hex_grid(2, 6, gap=gap))
        configs.append(make_hex_grid(6, 2, gap=gap))
        configs.append(make_ring(gap=gap))
        configs.append(make_ring12(gap=gap))
        configs.append(make_compact_cluster(gap=gap))
        configs.append(make_diamond(gap=gap))
        configs.append(make_trapezoid(gap=gap))
        configs.append(make_spiral(gap=gap))

    # Add rotated variants
    for base_cfg in configs[:10]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
        rotated2 = base_cfg.copy()
        rotated2[:, 2] = 15.0
        configs.append(rotated2)

    outer_rots = [0.0, 30.0, 15.0, 45.0]

    # Phase 1: Evaluate initial configs
    for cfg in configs:
        if time_left() < 30:
            break
        for orot in outer_rots:
            if time_left() < 30:
                break
            init_params = np.concatenate((cfg.flatten(), [orot]))
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()

    # Phase 2: Nelder-Mead optimization from best configs
    if best_params is not None and time_left() > 50:
        # Sort configs by initial L and try top ones
        candidate_params = []
        for cfg in configs[:20]:
            for orot in outer_rots[:2]:
                init_params = np.concatenate((cfg.flatten(), [orot]))
                if is_valid(init_params):
                    L_val = get_L(init_params)
                    candidate_params.append((L_val, init_params))
        
        candidate_params.sort(key=lambda x: x[0])
        
        for L_val, init_params in candidate_params[:8]:
            if time_left() < 30:
                break
            try:
                res_nm = minimize(
                    objective_combined,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                if is_valid(res_nm.x):
                    L_new = get_L(res_nm.x)
                    if L_new < best_L:
                        best_L = L_new
                        best_params = res_nm.x.copy()
            except Exception:
                pass

    # Phase 3: Differential evolution for global search
    if time_left() > 40:
        try:
            def de_objective(x):
                params = np.concatenate((x, [0.0]))
                if not is_valid(params):
                    return 100.0
                return objective_combined(params)
            
            bounds = [(-5, 5)] * (2*N) + [(-180, 180)] * N
            bounds[3*N - 1] = (-60, 60)  # outer rotation
            
            res_de = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if res_de.success or res_de.fun < best_L:
                x_de = res_de.x
                params_de = np.concatenate((x_de, [0.0]))
                if is_valid(params_de):
                    L_de = get_L(params_de)
                    if L_de < best_L:
                        best_L = L_de
                        best_params = params_de.copy()
        except Exception:
            pass

    # Phase 4: Simulated annealing
    if best_params is not None and time_left() > 35:
        current_params = best_params.copy()
        current_L = best_L
        T_start = 0.12
        T_end = 0.0003
        n_sa_iter = 1000
        sa_start_time = time.time()
        sa_budget = min(35, time_left() - 20)
        step_scale = 1.0
        accepted = 0
        total = 0

        for it in range(n_sa_iter):
            if time.time() - sa_start_time > sa_budget:
                break
            T = T_start * (T_end / T_start) ** (it / max(1, n_sa_iter - 1))
            total += 1

            if total > 50:
                acc_rate = accepted / total
                if acc_rate < 0.1:
                    step_scale *= 1.3
                elif acc_rate > 0.6:
                    step_scale *= 0.8
                step_scale = max(0.3, min(3.0, step_scale))

            new_params = current_params.copy()
            n_move = random.choice([1, 2, 3])
            base_step = 0.2 * step_scale
            for _ in range(n_move):
                idx = random.randint(0, N - 1)
                new_params[3*idx] += random.gauss(0, base_step)
                new_params[3*idx + 1] += random.gauss(0, base_step)
                new_params[3*idx + 2] += random.gauss(0, 3.0 * step_scale)
            new_params[3*N] += random.gauss(0, 1.0 * step_scale)

            if not is_valid(new_params):
                continue

            new_L = get_L(new_params)
            delta = new_L - current_L

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_params = new_params
                current_L = new_L
                accepted += 1
                if new_L < best_L:
                    best_L = new_L
                    best_params = new_params.copy()

    # Phase 5: Nelder-Mead polish after SA
    if best_params is not None and time_left() > 25:
        for restart in range(4):
            if time_left() < 10:
                break
            try:
                init_p = best_params.copy()
                if restart > 0:
                    for i in range(N):
                        init_p[3*i] += random.gauss(0, 0.04)
                        init_p[3*i+1] += random.gauss(0, 0.04)
                        init_p[3*i+2] += random.gauss(0, 0.8)
                    init_p[3*N] += random.gauss(0, 0.4)
                
                res_nm = minimize(
                    objective_combined,
                    init_p,
                    method='Nelder-Mead',
                    options={'maxiter': 600, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                if is_valid(res_nm.x):
                    L_val = get_L(res_nm.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_nm.x.copy()
            except Exception:
                pass

    # Phase 6: Interleaved outer rotation + position optimization
    if best_params is not None and time_left() > 20:
        for _round in range(3):
            if time_left() < 10:
                break
            try:
                xys_rots = best_params[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=3000)
                if L_opt < best_L:
                    best_L = L_opt
                    best_params = best_params.copy()
                    best_params[3*N] = rot_opt
            except Exception:
                pass
            
            if time_left() < 10:
                break
            try:
                res_final = minimize(
                    objective_combined,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if is_valid(res_final.x):
                    L_val = get_L(res_final.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_final.x.copy()
            except Exception:
                pass

    # Phase 7: Final high-precision polish
    if best_params is not None and time_left() > 8:
        try:
            res_final2 = minimize(
                objective_combined,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if is_valid(res_final2.x):
                L_val = get_L(res_final2.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final2.x.copy()
        except Exception:
            pass

    # Phase 8: Final outer rotation optimization
    if best_params is not None and time_left() > 4:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=5000)
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass

    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0

    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    return inner, center, side, angle
