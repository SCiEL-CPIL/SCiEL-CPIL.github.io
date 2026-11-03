import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]
SQRT3 = math.sqrt(3)


def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
        ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx * dx + dy * dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _UX * c1 - _UY * s1
        v1y = h1[1] + _UX * s1 + _UY * c1
        v2x = h2[0] + _UX * c2 - _UY * s2
        v2y = h2[1] + _UX * s2 + _UY * c2
        en1x = _ENX * c1 - _ENY * s1
        en1y = _ENX * s1 + _ENY * c1
        en2x = _ENX * c2 - _ENY * s2
        en2y = _ENX * s2 + _ENY * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / SQRT3) * max_projs
            return np.max(side_vals)

        # Vectorized coarse scan
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        best_per_angle = np.max(side_vals, axis=1)
        best_idx = np.argmin(best_per_angle)
        best_side = best_per_angle[best_idx]
        best_phi = float(best_idx * 0.5)

        # Fine tune
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i + 1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Smooth surrogate objective for gradient-based optimization
    def smooth_surrogate_objective(params, beta=50.0):
        """
        Smooth approximation of the min-outer-side objective using log-sum-exp.
        Also includes a penalty for overlapping hexagons.
        """
        pos = params[0:2*N].reshape(N, 2)
        ang_deg = params[2*N:3*N]
        ang_rad = np.radians(ang_deg)
        
        # Check for overlaps (penalty)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], ang_rad[i])
                h2 = (pos[j, 0], pos[j, 1], ang_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Compute penetration depth as penalty
                    dx = h1[0] - h2[0]
                    dy = h1[1] - h2[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Minimum distance between hexagon centers when touching is ~2*sqrt(3)
                    min_dist = 2 * SQRT3
                    if dist < min_dist:
                        overlap_penalty += (min_dist - dist) ** 2 * 100.0
        
        if overlap_penalty > 0:
            return overlap_penalty + 100.0
        
        # Compute smooth outer side
        all_verts = get_all_vertices(pos, ang_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Try a few orientations and use smooth max
        best_smooth_side = float('inf')
        for phi_deg in [0, 10, 20, 30, 40, 50]:
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            
            # Smooth max using log-sum-exp
            max_projs_smooth = np.zeros(6)
            for k in range(6):
                vals = abs_proj[:, k]
                max_val = np.max(vals)
                log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
                max_projs_smooth[k] = log_sum_exp / beta
            
            side_vals = (2.0 / SQRT3) * max_projs_smooth
            candidate = np.max(side_vals)
            if candidate < best_smooth_side:
                best_smooth_side = candidate
        
        return best_smooth_side

    def smooth_surrogate_gradient(params, beta=50.0):
        """Numerical gradient of smooth surrogate (for debugging/verification)"""
        eps = 1e-6
        grad = np.zeros_like(params)
        base = smooth_surrogate_objective(params, beta)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (smooth_surrogate_objective(params_plus, beta) - base) / eps
        return grad

    # Honeycomb initial configuration
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Multi-start SA with different initial configurations
    initial_configs = []

    # Config 1: Standard honeycomb
    initial_configs.append((base_positions.copy(), base_angles.copy()))

    # Config 2: Slightly perturbed honeycomb
    for trial in range(5):
        pos = base_positions.copy()
        ang = base_angles.copy()
        for i in range(N):
            pos[i, 0] += random.gauss(0, 0.4)
            pos[i, 1] += random.gauss(0, 0.4)
            ang[i] += random.gauss(0, 15.0)
        initial_configs.append((pos, ang))

    # Config 3: Ring arrangement
    ring_pos = np.zeros((N, 2))
    ring_ang = np.zeros(N)
    for i in range(N):
        angle = 2 * math.pi * i / N
        r = 3.5
        ring_pos[i, 0] = r * math.cos(angle)
        ring_pos[i, 1] = r * math.sin(angle)
        ring_ang[i] = angle * 180 / math.pi
    initial_configs.append((ring_pos, ring_ang))

    # Config 4: Two rows
    row_pos = np.zeros((N, 2))
    row_ang = np.zeros(N)
    for i in range(6):
        row_pos[i, 0] = (i - 2.5) * SQRT3
        row_pos[i, 1] = -1.5
        row_pos[i + 6, 0] = (i - 2.5) * SQRT3 + SQRT3 / 2
        row_pos[i + 6, 1] = 1.5
    initial_configs.append((row_pos, row_ang))

    # Config 5: Compact cluster with rotations
    cluster_pos = np.zeros((N, 2))
    cluster_ang = np.zeros(N)
    grid_positions = [
        (0, 0), (SQRT3, 0), (2*SQRT3, 0),
        (SQRT3/2, 1.5), (3*SQRT3/2, 1.5), (5*SQRT3/2, 1.5),
        (0, 3.0), (SQRT3, 3.0), (2*SQRT3, 3.0),
        (-SQRT3/2, 1.5), (-3*SQRT3/2, 1.5), (3*SQRT3, 0),
    ]
    for idx, (px, py) in enumerate(grid_positions):
        cluster_pos[idx, 0] = px
        cluster_pos[idx, 1] = py
        cluster_ang[idx] = random.choice([0, 30, 60, 90, 120])
    initial_configs.append((cluster_pos, cluster_ang))

    # Evaluate each initial config
    for pos, ang in initial_configs:
        if not is_valid_configuration(pos, ang):
            scale = 0.8
            pos_fixed = pos * scale
            if is_valid_configuration(pos_fixed, ang):
                pos = pos_fixed
            else:
                continue

        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        if side < best_side:
            best_side = side
            best_positions = pos.copy()
            best_angles = ang.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    # SA phase with the best initial config
    sa_time_budget = DEADLINE * 0.50
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)

    T = 0.3
    T_min = 0.0001
    cooling = 0.99997
    step_size = 0.25
    step_decay = 0.99999

    step_count = 0
    max_steps = 400000
    accept_count = 0
    total_count = 0

    def generate_move(current_pos, current_ang, step_sz):
        r = random.random()
        if r < 0.6:
            idxs = [random.randint(0, N - 1)]
        elif r < 0.85:
            idxs = random.sample(range(N), 2)
        else:
            idxs = random.sample(range(N), 3)

        new_pos = current_pos.copy()
        new_ang = current_ang.copy()

        for idx in idxs:
            dx = random.gauss(0, step_sz)
            dy = random.gauss(0, step_sz)
            da = random.gauss(0, 5.0)
            new_pos[idx, 0] += dx
            new_pos[idx, 1] += dy
            new_ang[idx] += da

        return new_pos, new_ang, idxs

    def check_validity(new_pos, new_ang, moved_idxs):
        angles_rad_new = np.radians(new_ang)
        for idx in moved_idxs:
            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
            for j in range(N):
                if j == idx:
                    continue
                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h_moved, h_other):
                    return False
        if len(moved_idxs) > 1:
            for i_idx in range(len(moved_idxs)):
                for j_idx in range(i_idx + 1, len(moved_idxs)):
                    i = moved_idxs[i_idx]
                    j = moved_idxs[j_idx]
                    h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad_new[i])
                    h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        return False
        return True

    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1

        new_positions, new_angles, moved_idxs = generate_move(current_positions, current_angles, step_size)

        valid = check_validity(new_positions, new_angles, moved_idxs)

        if not valid:
            T *= cooling
            if T < T_min:
                T = T_min
            step_size *= step_decay
            if step_size < 0.005:
                step_size = 0.005
            continue

        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)

        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            accept_count += 1
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            delta = new_side - current_side
            if delta < 0:
                delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy

        if total_count % 500 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.1:
                step_size *= 0.8
                if step_size < 0.005:
                    step_size = 0.005
            elif acc_rate > 0.5:
                step_size *= 1.2
                if step_size > 1.0:
                    step_size = 1.0
            accept_count = 0
            total_count = 0

        T *= cooling
        if T < T_min:
            T = T_min
        step_size *= step_decay
        if step_size < 0.005:
            step_size = 0.005

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)

    if remaining_time > 1.5:
        # Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side_cd, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side_cd
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()

            t_start = time.time()
            step = 0.05
            min_step = 0.001

            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_cd.copy()
                            new_ang = best_ang_cd.copy()

                            if dim == 0:
                                new_pos[idx, 0] += direction * step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * step
                            else:
                                new_ang[idx] += direction * step

                            angles_rad_new = np.radians(new_ang)
                            valid = True
                            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                            for j in range(N):
                                if j == idx:
                                    continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break

                            if not valid:
                                continue

                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)

                            if new_side < best_side_cd - 1e-10:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break

                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.8
                    continue
                break

            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.3, 15.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)

        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

        # NEW: Smooth Surrogate Gradient-Based Polish using L-BFGS
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
            def smooth_objective_for_lbfgs(params):
                """
                Objective for L-BFGS: smooth surrogate + overlap penalty.
                """
                pos = params[0:2*N].reshape(N, 2)
                ang_deg = params[2*N:3*N]
                
                # Quick validity check
                if not is_valid_configuration(pos, ang_deg):
                    # Return large penalty
                    return 1000.0
                
                # Use smooth surrogate
                return smooth_surrogate_objective(params, beta=30.0)

            # Initialize params from current best
            init_params = np.zeros(3 * N)
            for i in range(N):
                init_params[3 * i] = best_positions[i, 0]
                init_params[3 * i + 1] = best_positions[i, 1]
                init_params[3 * i + 2] = best_angles[i]

            try:
                res = minimize(smooth_objective_for_lbfgs, init_params, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                                        'maxfun': 2000})

                if res.fun < 100.0:  # Check if result is reasonable (not penalized)
                    new_pos = res.x[0:2*N].reshape(N, 2)
                    new_ang = res.x[2*N:3*N]

                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass

        # Nelder-Mead Polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 4.0:
            def polish_objective(params):
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = params[3 * i]
                    pos[i, 1] = params[3 * i + 1]
                    ang[i] = params[3 * i + 2]

                angles_rad = np.radians(ang)
                for i in range(N):
                    for j in range(i + 1, N):
                        h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                        h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return 1e6

                side, phi, cx, cy = compute_min_outer_side(pos, ang)
                return side

            init_params = np.zeros(3 * N)
            for i in range(N):
                init_params[3 * i] = best_positions[i, 0]
                init_params[3 * i + 1] = best_positions[i, 1]
                init_params[3 * i + 2] = best_angles[i]

            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})

                if res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3 * i]
                        new_pos[i, 1] = res.x[3 * i + 1]
                        new_ang[i] = res.x[3 * i + 2]

                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass

        # Final high-precision coordinate descent polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            def final_polish(pos, ang, time_limit):
                current_side_fp, _, _, _ = compute_min_outer_side(pos, ang)
                best_side_fp = current_side_fp
                best_pos_fp = pos.copy()
                best_ang_fp = ang.copy()

                t_start = time.time()
                step = 0.01
                min_step = 0.0001

                while time.time() - t_start < time_limit and step > min_step:
                    improved = False
                    for idx in range(N):
                        for dim in range(3):
                            for direction in [1, -1]:
                                new_pos = best_pos_fp.copy()
                                new_ang = best_ang_fp.copy()

                                if dim == 0:
                                    new_pos[idx, 0] += direction * step
                                elif dim == 1:
                                    new_pos[idx, 1] += direction * step
                                else:
                                    new_ang[idx] += direction * step

                                angles_rad_new = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                for j in range(N):
                                    if j == idx:
                                        continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid = False
                                        break

                                if not valid:
                                    continue

                                new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)

                                if new_side < best_side_fp - 1e-13:
                                    best_pos_fp = new_pos
                                    best_ang_fp = new_ang
                                    best_side_fp = new_side
                                    improved = True
                                    break

                            if time.time() - t_start > time_limit:
                                return best_side_fp, best_pos_fp, best_ang_fp
                        else:
                            continue
                        break
                    else:
                        step *= 0.7
                        continue
                    break

                return best_side_fp, best_pos_fp, best_ang_fp

            fp_time = min(remaining_time * 0.5, 15.0)
            fp_side, fp_pos, fp_ang = final_polish(best_positions, best_angles, fp_time)

            if fp_side < best_side:
                best_positions = fp_pos
                best_angles = fp_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
