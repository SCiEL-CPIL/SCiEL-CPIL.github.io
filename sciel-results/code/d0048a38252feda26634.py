import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution

N = 12
DEADLINE = 148.0

# Precompute unit hexagon vertices and edge normals
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
    random.seed(12345)
    np.random.seed(12345)

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
        phi_vals = np.radians(np.arange(0, 60, 0.25))
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
        best_phi = float(best_idx * 0.25)

        # Fine tune
        try:
            lo = max(0.0, best_phi - 0.25)
            hi = min(60.0, best_phi + 0.25)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-12})
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

    # Evaluate base honeycomb
    if is_valid_configuration(base_positions, base_angles):
        side, phi, cx, cy = compute_min_outer_side(base_positions, base_angles)
        best_side = side
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_phi = phi
        best_cx = cx
        best_cy = cy

    # Differential Evolution with smooth objective
    def de_objective(params):
        """Objective: outer side length + smooth collision penalty."""
        pos = params[:2 * N].reshape(N, 2)
        ang = params[2 * N:]

        # Smooth collision penalty using log-sum-exp
        angles_rad = np.radians(ang)
        log_penalties = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > 4.5:
                    continue
                a1, a2 = angles_rad[i], angles_rad[j]
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)
                v1x = pos[i, 0] + _UX * c1 - _UY * s1
                v1y = pos[i, 1] + _UX * s1 + _UY * c1
                v2x = pos[j, 0] + _UX * c2 - _UY * s2
                v2y = pos[j, 1] + _UX * s2 + _UY * c2
                en1x = _ENX * c1 - _ENY * s1
                en1y = _ENX * s1 + _ENY * c1
                en2x = _ENX * c2 - _ENY * s2
                en2y = _ENX * s2 + _ENY * c2

                # Compute separation for all 12 edge normals
                separations = []
                for i_e in range(6):
                    nx, ny = en1x[i_e], en1y[i_e]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    sep = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    separations.append(sep)
                for i_e in range(6):
                    nx, ny = en2x[i_e], en2y[i_e]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    sep = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    separations.append(sep)

                # Log-sum-exp for smooth max
                max_sep = max(separations)
                if max_sep > 0:
                    log_pen = max_sep + math.log(sum(math.exp(s - max_sep) for s in separations))
                    log_penalties.append(log_pen)

        penalty = 0.0
        if log_penalties:
            max_lp = max(log_penalties)
            penalty = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_penalties))
            penalty *= 5.0

        # Compute outer side
        try:
            side, _, _, _ = compute_min_outer_side(pos, ang)
        except Exception:
            side = 20.0

        return side + penalty

    # DE bounds
    bounds = [(-5.0, 5.0)] * (2 * N) + [(-180.0, 180.0)] * N

    # Run DE with time check
    def time_check():
        return time.time() - t0 > DEADLINE - 15.0

    try:
        de_result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=500,
            popsize=25,
            tol=1e-8,
            polish=False,
            callback=lambda xk, convergence: convergence < 0.5 and not time_check(),
            init='sobol',
        )
    except Exception:
        de_result = None

    if de_result is not None and de_result.fun < best_side + 5.0:
        new_pos = de_result.x[:2 * N].reshape(N, 2)
        new_ang = de_result.x[2 * N:]
        if is_valid_configuration(new_pos, new_ang):
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
            if new_side < best_side:
                best_side = new_side
                best_positions = new_pos
                best_angles = new_ang
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy

    # SA phase
    sa_time_budget = min(DEADLINE - 20.0 - (time.time() - t0), 60.0)
    if sa_time_budget < 5.0:
        sa_time_budget = 5.0

    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)

    T = 0.2
    T_min = 0.0001
    cooling = 0.99998
    step_size = 0.2
    step_decay = 0.999995

    def generate_move(cur_pos, cur_ang, step_sz):
        r = random.random()
        if r < 0.6:
            idxs = [random.randint(0, N - 1)]
        elif r < 0.85:
            idxs = random.sample(range(N), 2)
        else:
            idxs = random.sample(range(N), 3)

        new_pos = cur_pos.copy()
        new_ang = cur_ang.copy()

        for idx in idxs:
            dx = random.gauss(0, step_sz)
            dy = random.gauss(0, step_sz)
            da = random.gauss(0, 4.0)
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

    step_count = 0
    while time.time() - t0 < t0 + sa_time_budget:
        step_count += 1

        new_positions, new_angles, moved_idxs = generate_move(current_positions, current_angles, step_size)
        valid = check_validity(new_positions, new_angles, moved_idxs)

        if not valid:
            T = max(T * cooling, T_min)
            step_size = max(step_size * step_decay, 0.003)
            continue

        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)

        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            delta = max(new_side - current_side, 0)
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy

        T = max(T * cooling, T_min)
        step_size = max(step_size * step_decay, 0.003)

        # Adaptive step size
        if step_count % 200 == 0:
            # Estimate acceptance rate from recent history - simplified
            if step_size < 0.01:
                step_size = min(step_size * 1.3, 0.5)
            elif random.random() < 0.1:
                step_size = max(step_size * 0.9, 0.003)

    # Gradient-based polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 10.0:
        def polish_objective(params):
            pos = params[:2 * N].reshape(N, 2)
            ang = params[2 * N:]

            if not is_valid_configuration(pos, ang):
                return 1e6

            side, _, _, _ = compute_min_outer_side(pos, ang)
            return side

        init_params = np.zeros(3 * N)
        for i in range(N):
            init_params[2 * i] = best_positions[i, 0]
            init_params[2 * i + 1] = best_positions[i, 1]
            init_params[2 * N + i] = best_angles[i]

        try:
            res = minimize(polish_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})

            if res.fun < best_side:
                new_pos = res.x[:2 * N].reshape(N, 2)
                new_ang = res.x[2 * N:]

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

    # Final high-precision coordinate descent
    if time.time() - t0 < DEADLINE - 2.0:
        final_step = 0.005
        for _ in range(5):
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()

                        if dim == 0:
                            new_pos[idx, 0] += direction * final_step
                        elif dim == 1:
                            new_pos[idx, 1] += direction * final_step
                        else:
                            new_ang[idx] += direction * final_step

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

                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break

                    if time.time() - t0 > DEADLINE - 2.0:
                        break
                if time.time() - t0 > DEADLINE - 2.0:
                    break
            if not improved:
                break

    # Final verification and recompute
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
