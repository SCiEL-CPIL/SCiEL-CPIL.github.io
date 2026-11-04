import numpy as np
import time
import math
import random
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0  # Safety margin below 150s kill

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with rotation rot_deg."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns separation distance (positive) if not overlapping, or negative penetration if overlapping.
    Uses Separating Axis Theorem for convex polygons.
    """
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

        if max1 < min2:
            gap = min2 - max1
            return gap
        if max2 < min1:
            gap = min1 - max2
            return gap

        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)

    return -min_overlap

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def smooth_max_abs_proj(points, outer_rot_deg, beta=10.0):
    """
    Smooth approximation of max(|proj|) using log-sum-exp.
    This makes the outer constraint differentiable.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T  # shape: (n_points, 6)
    abs_proj = np.abs(proj)    # shape: (n_points, 6)
    
    # Smooth max over all entries using log-sum-exp
    flat = abs_proj.flatten()
    scaled = flat / beta
    max_s = np.max(scaled)
    if max_s > 500:
        return np.max(flat)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def smooth_overlap_penalty(xys_rots, beta=25.0):
    """
    Compute smooth overlap penalty using log-sum-exp for numerical stability.
    Returns a smooth approximation of the sum of squared penetrations.
    """
    centers = xys_rots[:, :2]

    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)

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
    if max_s > 500:
        return np.sum(p_arr ** 2)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

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

        # Re-center to remove translation invariance
        centers = xys_rots[:, :2]
        mean_center = np.mean(centers, axis=0)
        xys_rots_rc = xys_rots.copy()
        xys_rots_rc[:, :2] -= mean_center

        # Smooth overlap penalty
        penalty = smooth_overlap_penalty(xys_rots_rc, beta=25.0)

        # Compute outer side length using smooth max for differentiability
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots_rc])
        # Use smooth max for the objective to get gradients
        smooth_max_proj = smooth_max_abs_proj(all_points, outer_rot, beta=10.0)
        L = (2.0 / np.sqrt(3.0)) * smooth_max_proj

        return L + 100.0 * penalty

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

    def make_circular_cluster(gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k + 30)
            pts.append([(sqrt3 + gap) * np.cos(ang), (sqrt3 + gap) * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k)
            pts.append([(2 * sqrt3 + gap) * np.cos(ang), (2 * sqrt3 + gap) * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_linear_chain(gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        for i in range(N):
            x = i * (sqrt3 + gap)
            y = 0.0
            if i % 2 == 1:
                y = 1.5 + gap * 0.5
            pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_hexagonal_2x6(gap=0.0):
        """Two rows of 6 hexagons, offset."""
        pts = []
        sqrt3 = np.sqrt(3)
        for r in range(2):
            for c in range(6):
                x = c * (sqrt3 + gap)
                y = r * (1.5 + gap * 0.5)
                if r % 2 == 1:
                    x += (sqrt3 + gap) / 2.0
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_hexagonal_6x2(gap=0.0):
        """Six rows of 2 hexagons."""
        pts = []
        sqrt3 = np.sqrt(3)
        for r in range(6):
            for c in range(2):
                x = c * (sqrt3 + gap)
                y = r * (1.5 + gap * 0.5)
                if r % 2 == 1:
                    x += (sqrt3 + gap) / 2.0
                pts.append([x, y, 0.0])
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

    def get_L(params):
        xys_rots = params[:3 * N].reshape(N, 3)
        outer_rot = params[3 * N]
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)

    def full_score(cfg, angle):
        params = np.concatenate((cfg.flatten(), [angle]))
        if not is_valid(params):
            return None
        return get_L(params)

    best_L = float('inf')
    best_params = None

    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_hexagonal_2x6(gap=gap))
    configs.append(make_hexagonal_6x2(gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring12(gap=gap))
    configs.append(make_circular_cluster(gap=gap))
    configs.append(make_linear_chain(gap=gap))

    # Add rotated variants
    for base_cfg in configs[:6]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)

    outer_rots = [0.0, 30.0, 15.0, 45.0, -15.0, -45.0]

    # Phase 1: Evaluate initial configurations
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 35:
            break

        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 35:
                break

            init_params = np.concatenate((cfg.flatten(), [orot]))

            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()

    # Phase 2: Simulated annealing for global exploration
    if time.time() - start_time < DEADLINE - 45:
        if best_params is not None:
            cfg = best_params[:3 * N].reshape(N, 3).copy()
            angle = best_params[3 * N]
            cur_score = best_L

            T = 0.8
            T_end = 0.003
            alpha = 0.9997

            while time.time() - start_time < DEADLINE - 45:
                r = random.random()

                if r < 0.50:
                    idx = random.randrange(N)
                    old_x, old_y = cfg[idx][0], cfg[idx][1]
                    step = 0.2 * (1.0 + 0.5 * T)
                    cfg[idx][0] += random.gauss(0, step)
                    cfg[idx][1] += random.gauss(0, step)
                elif r < 0.70:
                    idx = random.randrange(N)
                    old_ang = cfg[idx][2]
                    cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
                elif r < 0.80:
                    old_angle = angle
                    angle += random.gauss(0, 2.0 + 2.0 * T)
                else:
                    idx1, idx2 = random.sample(range(N), 2)
                    old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
                    old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
                    step = 0.15 * (1.0 + 0.5 * T)
                    cfg[idx1][0] += random.gauss(0, step)
                    cfg[idx1][1] += random.gauss(0, step)
                    cfg[idx2][0] += random.gauss(0, step)
                    cfg[idx2][1] += random.gauss(0, step)

                new_score = full_score(cfg, angle)

                if new_score is not None:
                    if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                        cur_score = new_score
                        if new_score < best_L:
                            best_L = new_score
                            best_params = np.concatenate((cfg.flatten(), [angle])).copy()
                    else:
                        if r < 0.50:
                            cfg[idx][0], cfg[idx][1] = old_x, old_y
                        elif r < 0.70:
                            cfg[idx][2] = old_ang
                        elif r < 0.80:
                            angle = old_angle
                        else:
                            cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                            cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                else:
                    if r < 0.50:
                        cfg[idx][0], cfg[idx][1] = old_x, old_y
                    elif r < 0.70:
                        cfg[idx][2] = old_ang
                    elif r < 0.80:
                        angle = old_angle
                    else:
                        cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                        cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2

                T *= alpha
                if T < T_end:
                    break

    # Phase 3: Differential Evolution for global optimization
    if best_params is not None and time.time() - start_time < DEADLINE - 25:
        try:
            bounds = [(-4, 4), (-4, 4)] * N + [(-180, 180)] * N + [(-90, 90)]

            def de_objective(params):
                if time.time() - start_time > DEADLINE - 10:
                    return 1e6
                return objective(params)

            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=80,
                tol=1e-6,
                seed=42,
                polish=True,
                init='sobol'
            )

            if is_valid(de_result.x):
                L_val = get_L(de_result.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = de_result.x.copy()
        except Exception:
            pass

    # Phase 4: Gradient-based refinement from best result
    if best_params is not None and time.time() - start_time < DEADLINE - 15:
        try:
            res_lbgfs = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 800,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 3000
                }
            )

            if is_valid(res_lbgfs.x):
                L_val = get_L(res_lbgfs.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_lbgfs.x.copy()
        except Exception:
            pass

    # Phase 5: Nelder-Mead for final polish
    if best_params is not None and time.time() - start_time < DEADLINE - 8:
        try:
            res_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 8000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )

            if is_valid(res_nm.x):
                L_val = get_L(res_nm.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_nm.x.copy()
        except Exception:
            pass

    # Phase 6: High-precision final polish with COBYLA
    if best_params is not None and time.time() - start_time < DEADLINE - 3:
        try:
            res_cobyla = minimize(
                objective,
                best_params,
                method='COBYLA',
                options={
                    'maxiter': 5000,
                    'rhobeg': 1e-5,
                    'catol': 1e-12
                }
            )

            if is_valid(res_cobyla.x):
                L_val = get_L(res_cobyla.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_cobyla.x.copy()
        except Exception:
            pass

    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    xys_rots = best_params[:3 * N].reshape(N, 3)
    outer_rot = best_params[3 * N]

    inner = []
    for x, y, rot in xys_rots:
        inner.append([x, y, rot])

    return inner, (0.0, 0.0), best_L, outer_rot
