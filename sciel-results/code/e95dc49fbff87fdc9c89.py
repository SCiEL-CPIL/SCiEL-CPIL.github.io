import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth(v1, v2):
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        if np.min(overlaps) <= 1e-9:
            return 0.0
        return np.min(overlaps)

    def inside_violation(inner_verts, outer_verts):
        outer_normals = get_normals(outer_verts)
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        p_inner_max = p_inner.max(axis=0)
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def lattice_config_3x4():
        """3x4 rectangular arrangement on hexagonal lattice.
        Hexagonal lattice basis: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
        Adjacent hexagons (edge-to-edge) have center distance sqrt(3).
        """
        a1 = np.array([np.sqrt(3), 0.0])
        a2 = np.array([np.sqrt(3)/2, 1.5])
        pos = []
        for i in range(3):
            for j in range(4):
                pos.append(i * a1 + j * a2)
        return np.array(pos)

    def lattice_config_hex_cluster():
        """Hexagonal cluster: center + 6 ring + 5 of 12 second ring.
        Ring 1 distance: sqrt(3), Ring 2 distance: 2*sqrt(3)
        """
        d1 = np.sqrt(3)
        d2 = 2 * np.sqrt(3)
        pos = [[0.0, 0.0]]
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        # Second ring: 12 spots at angles 0,30,60,...,330
        # Pick 5 that form a compact shape: 0, 60, 120, 180, 240
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        return np.array(pos)

    def lattice_config_2x6():
        """2x6 arrangement on hexagonal lattice."""
        a1 = np.array([np.sqrt(3), 0.0])
        a2 = np.array([np.sqrt(3)/2, 1.5])
        pos = []
        for i in range(2):
            for j in range(6):
                pos.append(i * a1 + j * a2)
        return np.array(pos)

    def lattice_config_hex13():
        """Full hexagonal cluster of 13, remove one to get 12."""
        d1 = np.sqrt(3)
        d2 = 2 * np.sqrt(3)
        pos = [[0.0, 0.0]]
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        for k in range(12):
            a = np.radians(30 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        # Remove one from second ring to get 12
        return np.array(pos[:12])

    def objective(params, alpha_rad, theta_rad, cx, cy):
        if time.time() > deadline:
            return 1e10
        positions = params[:2 * N].reshape(N, 2)
        s = params[2 * N]
        if s < 2.5:
            return 1e6
        outer_verts = hex_vertices(cx, cy, s, theta_rad)
        total_penalty = 0.0
        inner_verts_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alpha_rad)
            inner_verts_list.append(verts)
            viol = inside_violation(verts, outer_verts)
            total_penalty += viol * 200.0
        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                total_penalty += pd * 200.0
        return s + total_penalty

    def objective_rigid(params, alpha_rad, theta_rad, base_pos):
        """Rigid body: only rotation of cluster, translation, outer size.
        params: [angle, tx, ty, s]
        """
        if time.time() > deadline:
            return 1e10
        angle, tx, ty, s = params
        if s < 2.5:
            return 1e6
        # Rotate base positions
        c, sn = np.cos(angle), np.sin(angle)
        R = np.array([[c, -sn], [sn, c]])
        positions = base_pos @ R.T + np.array([tx, ty])
        outer_verts = hex_vertices(0, 0, s, theta_rad)
        total_penalty = 0.0
        inner_verts_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alpha_rad)
            inner_verts_list.append(verts)
            viol = inside_violation(verts, outer_verts)
            total_penalty += viol * 200.0
        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                total_penalty += pd * 200.0
        return s + total_penalty

    best_s = np.inf
    best_result = None

    lattice_configs = {
        '3x4': lattice_config_3x4(),
        'hex_cluster': lattice_config_hex_cluster(),
        '2x6': lattice_config_2x6(),
        'hex13': lattice_config_hex13(),
    }

    # First pass: rigid body optimization for each lattice config
    for name, base_pos in lattice_configs.items():
        if time.time() > deadline - 30:
            break
        # Try different inner rotations and outer rotations
        for alpha_deg in [0, 30]:
            if time.time() > deadline - 30:
                break
            alpha_rad = np.radians(alpha_deg)
            for theta_deg in [0, 30]:
                if time.time() > deadline - 30:
                    break
                theta_rad = np.radians(theta_deg)
                # Initial guess for rigid body
                # Estimate bounding size
                extent = np.max(base_pos, axis=0) - np.min(base_pos, axis=0)
                init_s = max(extent) / 2 + 2.0
                x0 = np.array([0.0, 0.0, 0.0, init_s])
                try:
                    res = minimize(
                        objective_rigid, x0,
                        args=(alpha_rad, theta_rad, base_pos),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7}
                    )
                    angle, tx, ty, s = res.x
                    c, sn = np.cos(angle), np.sin(angle)
                    R = np.array([[c, -sn], [sn, c]])
                    positions = base_pos @ R.T + np.array([tx, ty])
                    # Validate
                    outer_verts = hex_vertices(0, 0, s, theta_rad)
                    valid = True
                    inner_verts_list = []
                    for i in range(N):
                        verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, alpha_rad)
                        inner_verts_list.append(verts)
                        if inside_violation(verts, outer_verts) > 1e-6:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i + 1, N):
                                if penetration_depth(inner_verts_list[i], inner_verts_list[j]) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid and s < best_s:
                        best_s = s
                        best_result = (positions, s, alpha_rad, theta_rad)
                except Exception:
                    pass

    # Second pass: full optimization starting from best rigid result
    if best_result is not None:
        positions0, s0, alpha0, theta0 = best_result
        if time.time() < deadline - 20:
            for alpha_deg in [0, 30]:
                if time.time() > deadline - 20:
                    break
                alpha_rad = np.radians(alpha_deg)
                for theta_deg in [0, 30]:
                    if time.time() > deadline - 20:
                        break
                    theta_rad = np.radians(theta_deg)
                    x0 = np.concatenate([positions0.flatten(), [s0]])
                    try:
                        res = minimize(
                            objective, x0,
                            args=(alpha_rad, theta_rad, 0.0, 0.0),
                            method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                        )
                        pos = res.x[:2 * N].reshape(N, 2)
                        s = res.x[2 * N]
                        outer_verts = hex_vertices(0, 0, s, theta_rad)
                        valid = True
                        inner_verts_list = []
                        for i in range(N):
                            verts = hex_vertices(pos[i, 0], pos[i, 1], 1.0, alpha_rad)
                            inner_verts_list.append(verts)
                            if inside_violation(verts, outer_verts) > 1e-6:
                                valid = False
                                break
                        if valid:
                            for i in range(N):
                                for j in range(i + 1, N):
                                    if penetration_depth(inner_verts_list[i], inner_verts_list[j]) > 1e-6:
                                        valid = False
                                        break
                                if not valid:
                                    break
                        if valid and s < best_s:
                            best_s = s
                            best_result = (pos, s, alpha_rad, theta_rad)
                    except Exception:
                        pass

    # Fallback if no valid solution
    if best_result is None:
        base_pos = lattice_config_3x4()
        positions = base_pos
        s = 5.0
        alpha_rad = 0.0
        theta_rad = 0.0
        best_result = (positions, s, alpha_rad, theta_rad)
        best_s = s

    final_pos, final_s, final_alpha, final_theta = best_result
    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(final_alpha)] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(final_theta)

    return inner, outer_center, outer_side, outer_angle_deg
