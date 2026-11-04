import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin below the 150 s wall-clock kill

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def get_hex_vertices_batch(xys_rots):
    """Batch vertices. xys_rots is (N,3) -> returns (N,6,2)."""
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    return np.stack((x, y), axis=-1)

def sat_overlap(v1, v2):
    """Penetration depth if overlapping (negative), else separation distance (positive)."""
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
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
    return -min_overlap if min_overlap < float('inf') else 0.0

def smooth_overlap_penalty(params, beta=30.0):
    """Smooth (log-sum-exp) approximation of the sum of squared SAT penetrations."""
    xys_rots = params[:3 * N].reshape(N, 3)
    centers = xys_rots[:, :2]
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    penetrations = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.2:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    if len(penetrations) == 0:
        return 0.0
    p_arr = np.array(penetrations)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 400:
        return np.sum(p_arr ** 2)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def outer_hex_side(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def is_valid(params, tol=1e-6):
    xys_rots = params[:3 * N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    for i in range(N):
        for j in range(i + 1, N):
            if sat_overlap(vertices_list[i], vertices_list[j]) < -tol:
                return False
    return True

def get_L(params):
    xys_rots = params[:3 * N].reshape(N, 3)
    outer_rot = params[3 * N]
    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    return outer_hex_side(all_points, outer_rot)

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        xys_rots = params[:3 * N].reshape(N, 3)
        outer_rot = params[3 * N]
        penalty = smooth_overlap_penalty(params, beta=30.0)
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        return L + 50.0 * penalty

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
        R = 2.5 + gap
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_grid_rotated(rows, cols, rot_offset=30.0, gap=0.05):
        base = make_hex_grid(rows, cols, rot_offset=0.0, gap=gap)
        base[:, 2] = rot_offset
        return base

    def make_staggered(rows, cols, rot_offset=0.0, gap=0.0):
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
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    best_L = float('inf')
    best_params = None

    def consider(params):
        nonlocal best_L, best_params
        if not is_valid(params):
            return
        L = get_L(params)
        if L < best_L:
            best_L = L
            best_params = params.copy()

    def run_nm(params, maxiter, xatol, fatol):
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': xatol,
                                    'fatol': fatol, 'adaptive': True})
            return res.x
        except Exception:
            return params

    def sa_perturb(params, iters, T0, T1, scale):
        cur = params.copy()
        cur_L = float('inf')
        if is_valid(cur):
            cur_L = get_L(cur)
        best_local = cur.copy()
        best_local_L = cur_L
        for it in range(iters):
            if time.time() - start_time > DEADLINE - 15:
                break
            cand = cur.copy()
            idx = random.randrange(3 * N + 1)
            if idx < 3 * N:
                cand[idx] += random.gauss(0.0, scale)
            else:
                cand[idx] += random.gauss(0.0, 0.5)
            if is_valid(cand):
                cand_L = get_L(cand)
            else:
                cand_L = float('inf')
            T = T0 * (T1 / T0) ** (it / max(1, iters))
            if cand_L < cur_L or random.random() < math.exp(-(cand_L - cur_L) / max(T, 1e-9)):
                cur = cand
                cur_L = cand_L
                if cur_L < best_local_L:
                    best_local_L = cur_L
                    best_local = cur.copy()
        return best_local

    # Diverse initial family
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring12(gap=gap))
    configs.append(make_grid_rotated(4, 3, rot_offset=30.0, gap=gap))
    configs.append(make_grid_rotated(3, 4, rot_offset=30.0, gap=gap))
    configs.append(make_grid_rotated(2, 6, rot_offset=30.0, gap=gap))
    configs.append(make_grid_rotated(6, 2, rot_offset=30.0, gap=gap))
    configs.append(make_staggered(4, 3, rot_offset=15.0, gap=gap))
    configs.append(make_staggered(3, 4, rot_offset=15.0, gap=gap))
    # Additional diverse starts
    configs.append(make_hex_grid(4, 3, gap=0.1))
    configs.append(make_hex_grid(3, 4, gap=0.1))
    configs.append(make_ring12(gap=0.1))
    configs.append(make_ring(gap=0.1))

    outer_rots = [0.0, 30.0, 15.0, 45.0, 7.5, 22.5]

    # Phase 1: Nelder-Mead per diverse start
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 40:
            break
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 40:
                break
            init_params = np.concatenate((cfg.flatten(), [orot]))
            consider(init_params)
            res_x = run_nm(init_params, 300, 1e-6, 1e-6)
            consider(res_x)

    # Phase 2: simulated-annealing escape from current best, then re-optimize
    if best_params is not None and time.time() - start_time < DEADLINE - 45:
        for _ in range(6):
            if time.time() - start_time > DEADLINE - 45:
                break
            sa_out = sa_perturb(best_params, 400, 0.8, 0.005, 0.4)
            consider(sa_out)
            res_x = run_nm(sa_out, 400, 1e-7, 1e-7)
            consider(res_x)

    # Phase 3: more SA + NM cycles
    if best_params is not None and time.time() - start_time < DEADLINE - 30:
        for _ in range(4):
            if time.time() - start_time > DEADLINE - 30:
                break
            sa_out = sa_perturb(best_params, 500, 0.5, 0.002, 0.25)
            consider(sa_out)
            res_x = run_nm(sa_out, 500, 1e-8, 1e-8)
            consider(res_x)

    # Phase 4: final tight polish on the global best
    if best_params is not None and time.time() - start_time < DEADLINE - 15:
        res_x = run_nm(best_params, 3000, 1e-10, 1e-10)
        consider(res_x)

    # Phase 5: ultra-tight final polish
    if best_params is not None and time.time() - start_time < DEADLINE - 5:
        res_x = run_nm(best_params, 5000, 1e-14, 1e-14)
        consider(res_x)

    # Fallback: a simple valid spread
    if best_params is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0

    xys_rots = best_params[:3 * N].reshape(N, 3)
    outer_rot = best_params[3 * N]
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    return inner, center, side, angle
