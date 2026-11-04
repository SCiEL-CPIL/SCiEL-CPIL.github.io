import numpy as np
import time

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi / 6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    """Exact minimum side length of a regular hexagon (center, rotation) enclosing the point set."""
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _min_enclosing_hex_side_smooth(composite_verts, outer_angle_deg, outer_center, beta=100.0):
    """Smooth approximation using log-sum-exp."""
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    abs_dots = np.abs(dots)
    # log-sum-exp for max
    max_val = np.max(abs_dots)
    log_sum_exp = max_val + np.log(np.sum(np.exp(abs_dots - max_val)))
    return 2 * log_sum_exp / np.sqrt(3)

def _all_inner_vertices(centers, angles):
    """All vertices of all inner hexagons, as (N*6, 2)."""
    all_verts = []
    for i, (cx, cy) in enumerate(centers):
        all_verts.append(_hex_vertices(cx, cy, angles[i]))
    return np.vstack(all_verts)

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    best_S = np.inf
    best_outer = np.array([0.0, 0.0, 0.0])
    best_centers = None
    best_angles = None
    best_n = 0

    # Axial to pixel for flat-top hexagons
    def _axial_to_pycord(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 3.0 / 2.0 * r
        return x, y

    # Generate hex grid configurations
    def _generate_hex_grid(radius):
        centers = []
        angles = []
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                x, y = _axial_to_pycord(q, r)
                centers.append((x, y))
                angles.append(0.0)
        return centers, angles

    # Try radii 1, 2, 3
    for radius in [1, 2, 3]:
        centers_list, angles_list = _generate_hex_grid(radius)
        n = len(centers_list)
        centers_arr = np.array(centers_list)
        angles_arr = np.array(angles_list)

        all_verts = _all_inner_vertices(centers_arr, angles_arr)

        for outer_angle in [0.0, 30.0, 15.0, 45.0, 60.0]:
            side = _min_enclosing_hex_side(all_verts, outer_angle, np.array([0.0, 0.0]))
            if side < best_S:
                best_S = side
                best_outer = np.array([0.0, 0.0, outer_angle])
                best_centers = centers_arr.copy()
                best_angles = angles_arr.copy()
                best_n = n

    # Also try the original flower pattern with 30 deg rotation
    centers_flower = [(0, 0)]
    angles_flower = [30.0]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        centers_flower.append((r * np.cos(angle), r * np.sin(angle)))
        angles_flower.append(30.0)
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        centers_flower.append((r * np.cos(angle), r * np.sin(angle)))
        angles_flower.append(30.0)

    centers_arr = np.array(centers_flower)
    angles_arr = np.array(angles_flower)
    all_verts = _all_inner_vertices(centers_arr, angles_arr)
    for outer_angle in [0.0, 30.0, 15.0, 45.0]:
        side = _min_enclosing_hex_side(all_verts, outer_angle, np.array([0.0, 0.0]))
        if side < best_S:
            best_S = side
            best_outer = np.array([0.0, 0.0, outer_angle])
            best_centers = centers_arr.copy()
            best_angles = angles_arr.copy()
            best_n = len(centers_arr)

    # Phase 1: Polish outer hexagon
    all_verts = _all_inner_vertices(best_centers, best_angles)

    def _outer_objective(vars):
        outer_cx, outer_cy, outer_angle = vars
        outer_center = np.array([outer_cx, outer_cy])
        return _min_enclosing_hex_side(all_verts, outer_angle, outer_center)

    if time.time() < DEADLINE - 10.0:
        try:
            from scipy.optimize import minimize
            x0 = best_outer.copy()
            res = minimize(_outer_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_outer = res.x.copy()
        except Exception:
            pass

    # Phase 2: Joint polish
    if time.time() < DEADLINE - 20.0:
        try:
            from scipy.optimize import minimize

            def _joint_objective(vars):
                outer_cx = vars[0]
                outer_cy = vars[1]
                outer_angle = vars[2]
                c = vars[3:3 + 2 * best_n].reshape(best_n, 2)
                verts = []
                for i in range(best_n):
                    verts.append(_hex_vertices(c[i][0], c[i][1], best_angles[i]))
                all_v = np.vstack(verts)
                return _min_enclosing_hex_side(all_v, outer_angle, np.array([outer_cx, outer_cy]))

            x0_joint = np.zeros(3 + 2 * best_n)
            x0_joint[0] = best_outer[0]
            x0_joint[1] = best_outer[1]
            x0_joint[2] = best_outer[2]
            for i in range(best_n):
                x0_joint[3 + 2 * i] = best_centers[i][0]
                x0_joint[4 + 2 * i] = best_centers[i][1]

            res = minimize(_joint_objective, x0_joint, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
            if time.time() < DEADLINE - 5.0 and res.fun < best_S:
                best_S = res.fun
                best_outer = res.x[:3].copy()
                best_centers = res.x[3:3 + 2 * best_n].reshape(best_n, 2).copy()
        except Exception:
            pass

    outer_cx, outer_cy, outer_angle = best_outer
    outer_center = (outer_cx, outer_cy)

    inner = []
    for i in range(best_n):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])

    side = best_S

    return inner, outer_center, side, outer_angle
