import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 15.0

    d = math.sqrt(3)

    # Generate centers: center + ring 1 (6) + ring 2 (12)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))

    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))

    base_centers = centers[:7]

    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    base_points = []
    for cx, cy in base_centers:
        for vx, vy in verts_rel:
            base_points.append((cx + vx, cy + vy))
    base_points_arr = np.array(base_points)

    ring2_points_list = []
    for cx, cy in ring2_centers:
        pts = []
        for vx, vy in verts_rel:
            pts.append((cx + vx, cy + vy))
        ring2_points_list.append(np.array(pts))

    all_combinations = list(itertools.combinations(range(12), 4))

    def calc_S(points_arr, theta):
        k_angles = np.arange(6) * math.pi / 3
        angles = theta + k_angles
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        dots = np.sum(normals[:, np.newaxis, :] * points_arr[np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        max_per_normal = np.max(abs_dots, axis=-1)
        max_proj = np.max(max_per_normal)
        return (2.0 / math.sqrt(3)) * max_proj

    best_S = float('inf')
    best_combo = None
    best_theta = 0.0
    top_candidates = []

    n_coarse = 100
    theta_coarse = np.linspace(0, math.pi / 3, n_coarse, endpoint=False)

    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break

        combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])

        k_angles = np.arange(6) * math.pi / 3
        angles = theta_coarse[:, np.newaxis] + k_angles[np.newaxis, :]
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

        dots = np.sum(normals[:, :, np.newaxis, :] * combo_points[np.newaxis, np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        max_per_normal = np.max(abs_dots, axis=-1)
        max_proj = np.max(max_per_normal, axis=-1)
        S_vals = (2.0 / math.sqrt(3)) * max_proj

        min_idx = np.argmin(S_vals)
        min_S = S_vals[min_idx]
        min_theta = theta_coarse[min_idx]

        top_candidates.append((min_S, combo, min_theta))

        if min_S < best_S:
            best_S = min_S
            best_combo = combo
            best_theta = min_theta

    top_candidates.sort(key=lambda x: x[0])

    refined_best_S = float('inf')
    refined_best_combo = None
    refined_best_theta = 0.0

    if time.time() < deadline - safety_margin:
        for S_coarse, combo, theta_init in top_candidates[:10]:
            if time.time() > deadline - safety_margin:
                break

            combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])

            def obj(theta):
                return calc_S(combo_points, theta)

            try:
                res = minimize_scalar(obj, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-8})
                if res.success:
                    S_opt = res.fun
                    theta_opt = res.x
                    if S_opt < refined_best_S:
                        refined_best_S = S_opt
                        refined_best_combo = combo
                        refined_best_theta = theta_opt
            except Exception:
                continue

    if refined_best_combo is not None:
        best_S = refined_best_S
        best_combo = refined_best_combo
        best_theta = refined_best_theta

    if best_combo is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    final_centers = list(base_centers)
    for idx in best_combo:
        final_centers.append(ring2_centers[idx])

    # Proper SAT-based overlap check
    def get_hex_verts(cx, cy, angle):
        verts = np.zeros((6, 2))
        for k in range(6):
            a = angle + k * math.pi / 3
            verts[k, 0] = cx + math.cos(a)
            verts[k, 1] = cy + math.sin(a)
        return verts

    def sat_overlap(v1, v2):
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def check_all_overlap(params):
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(11)]
        for i in range(11):
            for j in range(i + 1, 11):
                if sat_overlap(verts_list[i], verts_list[j]):
                    return True
        return False

    def compute_S(params):
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        max_proj = 0.0
        for i in range(11):
            c = math.cos(thetas[i])
            s = math.sin(thetas[i])
            rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
            ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
            px = rx + xs[i]
            py = ry + ys[i]
            for k in range(6):
                ang = k * math.pi / 3
                nx = math.cos(ang)
                ny = math.sin(ang)
                proj = np.max(np.abs(px * nx + py * ny))
                if proj > max_proj:
                    max_proj = proj
        return (2.0 / math.sqrt(3)) * max_proj

    # Local optimization
    best_inner_data = None
    best_outer_angle = 0.0

    if time.time() < deadline - safety_margin:
        init_params = []
        for cx, cy in final_centers:
            init_params.extend([cx, cy, 0.0])
        init_params = np.array(init_params)

        def objective(params):
            if time.time() > deadline - safety_margin:
                return 1e10
            xs = params[0::3]
            ys = params[1::3]
            thetas = params[2::3]

            # SAT overlap check
            verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(11)]
            penalty = 0.0
            for i in range(11):
                for j in range(i + 1, 11):
                    if sat_overlap(verts_list[i], verts_list[j]):
                        # Compute minimum translation to separate (approximate by center distance)
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = math.sqrt(dx * dx + dy * dy)
                        penalty += max(0, math.sqrt(3) - dist) ** 2 * 100.0

            S = compute_S(params)
            return S + penalty

        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})

            if res.success or res.fun < best_S:
                final_params = res.x
                if not check_all_overlap(final_params):
                    new_S = compute_S(final_params)
                    if new_S < best_S:
                        best_S = new_S
                        best_inner_data = [[final_params[3*i], final_params[3*i+1], math.degrees(final_params[3*i+2])] for i in range(11)]
        except Exception:
            pass

    if best_inner_data is not None:
        return best_inner_data, (0.0, 0.0), best_S, best_outer_angle

    inner_rows = []
    for cx, cy in final_centers:
        inner_rows.append([cx, cy, 0.0])

    return inner_rows, (0.0, 0.0), best_S, math.degrees(best_theta)
