import numpy as np
import time
import math
import random
from scipy.optimize import minimize, least_squares

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
    return -min_overlap

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

def make_honeycomb_cluster(rot_offset=0.0):
    """Create a dense honeycomb cluster of 12 unit hexagons with optimal local packing.
    
    The honeycomb lattice has two types of positions per row. For 12 hexagons,
    we use a compact arrangement: 4 rows of 3, or 3 rows of 4, or other topologies.
    We'll try several topologies and return the one with smallest bounding hexagon.
    """
    sqrt3 = np.sqrt(3)
    
    def compute_L_for_cluster(pts):
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in pts]
        all_points = np.vstack(vertices_list)
        best_L = float('inf')
        for rot in np.linspace(0, 60, 360, endpoint=False):
            L = outer_hex_side(all_points, rot)
            if L < best_L:
                best_L = L
        return best_L
    
    best_L = float('inf')
    best_pts = None
    
    # Try different honeycomb topologies
    topologies = [
        (4, 3),  # 4 rows x 3 cols
        (3, 4),  # 3 rows x 4 cols
        (2, 6),  # 2 rows x 6 cols
        (6, 2),  # 6 rows x 2 cols
    ]
    
    for rows, cols in topologies:
        pts = make_hex_grid(rows, cols, rot_offset=rot_offset, gap=0.0)
        if len(pts) == N:
            L = compute_L_for_cluster(pts)
            if L < best_L:
                best_L = L
                best_pts = pts.copy()
    
    # Also try a 1+6+5 ring topology (center + 6 neighbors + 5 more)
    ring_pts = make_ring12(gap=0.0, radial=True)
    L = compute_L_for_cluster(ring_pts)
    if L < best_L:
        best_L = L
        best_pts = ring_pts.copy()
    
    if best_pts is None:
        best_pts = make_hex_grid(4, 3, rot_offset=rot_offset, gap=0.0)
    
    return best_pts

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

def make_ring12(gap=0.0, radial=True):
    """12 hexagons in a ring, optionally oriented radially."""
    pts = []
    R = 2.6 + gap
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        angle_deg = math.degrees(theta) if radial else 0.0
        pts.append([x, y, angle_deg])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def is_valid(params, tol=1e-6):
    xys_rots = params[:3 * N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    for i in range(N):
        for j in range(i + 1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < -tol:
                return False
    return True

def compute_penetrations(params):
    xys_rots = params[:3 * N].reshape(N, 3)
    centers = xys_rots[:, :2]
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    penetrations = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 3.5:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                penetrations.append(d_sat)
            else:
                penetrations.append(1.0)
    return np.array(penetrations)

def objective_combined(params, penalty_weight=200.0):
    xys_rots = params[:3 * N].reshape(N, 3)
    outer_rot = params[3 * N]
    centers = xys_rots[:, :2]
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 3.5:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penalty += d_sat * d_sat
    all_points = np.vstack(vertices_list)
    L = outer_hex_side(all_points, outer_rot)
    return L + penalty_weight * penalty

def residuals_least_squares(params):
    penetrations = compute_penetrations(params)
    residuals = np.maximum(0.0, -penetrations)
    return residuals

def polish_local(params, time_budget, maxiter):
    """Nelder-Mead polish with overlap penalty, returns best valid result."""
    try:
        res = minimize(
            objective_combined,
            params,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        if is_valid(res.x):
            xys_rots = res.x[:3 * N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L = outer_hex_side(all_points, res.x[3 * N])
            return res.x.copy(), L
    except Exception:
        pass
    return None, float('inf')

def anneal(params, start_temp, end_temp, n_iter, time_budget):
    """Simulated annealing for global exploration over inner positions/rotations."""
    best_params = params.copy()
    best_L = float('inf')
    current = params.copy()
    current_L = objective_combined(current)

    # initial valid L
    if is_valid(current):
        xys_rots = current[:3 * N].reshape(N, 3)
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        best_L = outer_hex_side(all_points, current[3 * N])
        current_L = best_L

    t0 = time.time()
    for it in range(n_iter):
        if time.time() - t0 > time_budget:
            break
        temp = start_temp * (end_temp / start_temp) ** (it / max(1, n_iter - 1))
        # propose move: pick random hexagon, nudge position and rotation
        idx = random.randint(0, N - 1)
        cand = current.copy()
        cand[idx * 3 + 0] += random.gauss(0, 0.3)
        cand[idx * 3 + 1] += random.gauss(0, 0.3)
        cand[idx * 3 + 2] += random.gauss(0, 15.0)
        cand_obj = objective_combined(cand)
        delta = cand_obj - current_L
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
            current = cand
            current_L = cand_obj
            if is_valid(cand):
                xys_rots = cand[:3 * N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L = outer_hex_side(all_points, cand[3 * N])
                if L < best_L:
                    best_L = L
                    best_params = cand.copy()
    return best_params, best_L

def construct():
    start_time = time.time()
    random.seed(7)
    np.random.seed(7)

    def time_left():
        return DEADLINE - (time.time() - start_time)

    best_L = float('inf')
    best_params = None

    def try_record(params):
        nonlocal best_L, best_params
        if is_valid(params):
            xys_rots = params[:3 * N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L = outer_hex_side(all_points, params[3 * N])
            if L < best_L:
                best_L = L
                best_params = params.copy()

    # Diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring12(gap=gap, radial=True))
    configs.append(make_ring12(gap=gap, radial=False))
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    # Add honeycomb cluster configurations with various rotations
    for rot_off in [0.0, 30.0, 15.0, 45.0]:
        hc = make_honeycomb_cluster(rot_offset=rot_off)
        configs.append(hc)

    outer_rots = [0.0, 30.0, 15.0]

    for cfg in configs:
        if time_left() < 25:
            break
        for orot in outer_rots:
            if time_left() < 25:
                break
            init_params = np.concatenate((cfg.flatten(), [orot]))
            try_record(init_params)

            # least_squares to remove overlaps
            if time_left() < 30:
                continue
            try:
                res_ls = least_squares(
                    residuals_least_squares, init_params, method='trf',
                    max_nfev=120, ftol=1e-7, gtol=1e-7
                )
                try_record(res_ls.x)
            except Exception:
                pass

    # Main loop: anneal + polish from best, exploring
    if best_params is not None and time_left() > 40:
        current_best = best_params.copy()
        current_best_L = best_L
        while time_left() > 40:
            # Simulated annealing exploration from current best
            anneal_budget = min(12.0, time_left() - 40)
            if anneal_budget > 5:
                ann_params, ann_L = anneal(
                    current_best, start_temp=0.8, end_temp=0.02,
                    n_iter=400, time_budget=anneal_budget
                )
                if ann_L < best_L:
                    best_L = ann_L
                    best_params = ann_params.copy()
                    current_best = ann_params.copy()
                    current_best_L = ann_L

            # Nelder-Mead polish
            if time_left() > 45:
                p, L = polish_local(current_best, time_budget=8.0, maxiter=120)
                if L < best_L:
                    best_L = L
                    best_params = p.copy()
                    current_best = p.copy()
                    current_best_L = L

            # Refine outer rotation
            if time_left() > 45:
                xys_rots = best_params[:3 * N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3 * N], n_samples=720)
                if L_opt < best_L:
                    best_L = L_opt
                    new_params = best_params.copy()
                    new_params[3 * N] = rot_opt
                    best_params = new_params
                    current_best = new_params.copy()
                    current_best_L = L_opt

    # Final high-precision polish
    if best_params is not None and time_left() > 20:
        p, L = polish_local(best_params, time_budget=15.0, maxiter=600)
        if L < best_L:
            best_L = L
            best_params = p.copy()

    # Final outer rotation optimization with finer grid
    if best_params is not None and time_left() > 5:
        xys_rots = best_params[:3 * N].reshape(N, 3)
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3 * N], n_samples=2880)
        if L_opt < best_L:
            best_L = L_opt
            new_params = best_params.copy()
            new_params[3 * N] = rot_opt
            best_params = new_params
        
        # Additional fine-tuning of outer rotation
        if time_left() > 2:
            xys_rots = best_params[:3 * N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            # Very fine local search around best rotation
            best_rot_local = best_params[3 * N]
            fine_rots = np.linspace(best_rot_local - 0.5, best_rot_local + 0.5, 1000)
            for rot in fine_rots:
                L = outer_hex_side(all_points, rot)
                if L < best_L:
                    best_L = L
                    new_params = best_params.copy()
                    new_params[3 * N] = rot
                    best_params = new_params

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
