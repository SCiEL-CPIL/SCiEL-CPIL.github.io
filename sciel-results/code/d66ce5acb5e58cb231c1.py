import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3)
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
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
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)

        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 3):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg

        for phi_deg in [best_phi - 2, best_phi - 1, best_phi, best_phi + 1, best_phi + 2]:
            if 0 <= phi_deg < 60:
                s = eval_phi(phi_deg)
                if s < best_side:
                    best_side = s
                    best_phi = phi_deg

        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-9, 'maxiter': 50})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Generate multiple initial configurations
    def make_honeycomb(perturb=0.0, seed=None):
        if seed is not None:
            random.seed(seed)
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
            if perturb > 0:
                pos[idx, 0] += random.gauss(0, perturb)
                pos[idx, 1] += random.gauss(0, perturb)
                ang[idx] = random.gauss(0, perturb * 5)
        return pos, ang

    def make_compact_ring(perturb=0.0, seed=None):
        if seed is not None:
            random.seed(seed)
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Center hexagon
        pos[0] = [0, 0]
        # 6 around center at distance sqrt(3)
        for i in range(1, 7):
            theta = i * np.pi / 3
            pos[i] = [sqrt3 * math.cos(theta), sqrt3 * math.sin(theta)]
        # 5 more in a second ring, placed to fill gaps
        for i in range(7, 12):
            theta = (i - 7) * 2 * np.pi / 5 + np.pi / 5
            r = 2.5 * sqrt3
            pos[i] = [r * math.cos(theta), r * math.sin(theta)]
            ang[i] = theta * 180 / np.pi
        if perturb > 0:
            for i in range(N):
                pos[i, 0] += random.gauss(0, perturb)
                pos[i, 1] += random.gauss(0, perturb)
                ang[i] += random.gauss(0, perturb * 5)
        return pos, ang

    def make_two_rows(perturb=0.0, seed=None):
        if seed is not None:
            random.seed(seed)
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Two rows of 6
        for i in range(6):
            pos[i] = [i * sqrt3, 0]
            pos[i+6] = [(i + 0.5) * sqrt3, 1.5]
        if perturb > 0:
            for i in range(N):
                pos[i, 0] += random.gauss(0, perturb)
                pos[i, 1] += random.gauss(0, perturb)
                ang[i] = random.gauss(0, perturb * 5)
        return pos, ang

    # SA function
    def run_sa(pos_init, ang_init, time_budget, T_init=0.4, step_init=0.3):
        current_positions = pos_init.copy()
        current_angles = ang_init.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

        T = T_init
        T_min = 0.0001
        cooling = 0.99995
        step_size = step_init
        step_decay = 0.99999
        accept_count = 0
        total_count = 0
        t_start = time.time()

        while time.time() - t_start < time_budget:
            total_count += 1
            idx = random.randint(0, N - 1)
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()

            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            da = random.gauss(0, 4.0)

            new_positions[idx, 0] += dx
            new_positions[idx, 1] += dy
            new_angles[idx] += da

            angles_rad_new = np.radians(new_angles)
            valid = True
            h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
            for j in range(N):
                if j == idx:
                    continue
                h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h_moved, h_other):
                    valid = False
                    break

            if not valid:
                T *= cooling
                if T < T_min: T = T_min
                step_size *= step_decay
                if step_size < 0.003: step_size = 0.003
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
                if delta < 0: delta = 0
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

            if total_count % 300 == 0:
                acc_rate = accept_count / max(1, total_count)
                if acc_rate < 0.08:
                    step_size *= 0.85
                    if step_size < 0.003: step_size = 0.003
                elif acc_rate > 0.5:
                    step_size *= 1.15
                    if step_size > 0.8: step_size = 0.8
                accept_count = 0
                total_count = 0

            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003

        return best_side, best_positions, best_angles, best_phi, best_cx, best_cy

    # Coordinate descent polish
    def polish_cd(pos, ang, time_limit, step=0.05, min_step=0.001):
        current_side, _, _, _ = compute_min_outer_side(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()
        t_start = time.time()

        while time.time() - t_start < time_limit and step > min_step:
            improved = False
            for idx in range(N):
                if time.time() - t_start > time_limit:
                    break
                for dim in range(3):
                    if time.time() - t_start > time_limit:
                        break
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
                            if j == idx: continue
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
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.75

        return best_side_cd, best_pos_cd, best_ang_cd

    # Smooth surrogate for L-BFGS-B
    def smooth_outer_side(pos, ang, beta=50.0):
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 3):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_val = np.max(abs_proj, axis=0)
            shifted = abs_proj - max_val[None, :]
            log_sum_exp = max_val + np.log(np.sum(np.exp(shifted), axis=0) / len(offsets))
            side_vals = (2.0 / sqrt3) * log_sum_exp
            side = np.max(side_vals)
            if side < best_side:
                best_side = side
                best_phi = phi_deg
        return best_side, best_phi, center[0], center[1]

    def smooth_objective(params, beta=50.0):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        angles_rad = np.radians(ang)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return 1e6
        side, phi, cx, cy = smooth_outer_side(pos, ang, beta)
        return side

    # Main search
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Phase 1: Multiple SA restarts
    elapsed = time.time() - t0
    sa_budget = DEADLINE - elapsed - 30.0  # reserve 30s for polish
    if sa_budget < 10.0:
        sa_budget = 10.0

    # Try up to 3 restarts if time permits
    inits = [
        make_honeycomb(perturb=0.0, seed=42),
        make_honeycomb(perturb=0.15, seed=123),
        make_compact_ring(perturb=0.1, seed=77),
    ]

    n_restarts = min(3, max(1, int(sa_budget / 30.0)))
    per_restart = sa_budget / n_restarts

    for r in range(n_restarts):
        if time.time() - t0 > DEADLINE - 25.0:
            break
        pos_init, ang_init = inits[r]
        # Verify initial config is valid
        if not is_valid_configuration(pos_init, ang_init):
            # Try to fix by scaling
            pos_init = pos_init * 0.95
            if not is_valid_configuration(pos_init, ang_init):
                continue

        t_start = time.time()
        b_side, b_pos, b_ang, b_phi, b_cx, b_cy = run_sa(
            pos_init, ang_init, per_restart - (time.time() - t_start))
        if b_side < best_side:
            best_side = b_side
            best_positions = b_pos
            best_angles = b_ang
            best_phi = b_phi
            best_cx = b_cx
            best_cy = b_cy

    # Phase 2: Polish
    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    remaining = DEADLINE - (time.time() - t0)

    # 2a: Coordinate descent
    if remaining > 8.0:
        cd_time = min(remaining * 0.35, 15.0)
        cd_side, cd_pos, cd_ang = polish_cd(best_positions, best_angles, cd_time, step=0.04, min_step=0.002)
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # 2b: Smooth surrogate + L-BFGS-B
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10.0:
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
        try:
            res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                           options={'maxiter': 150, 'gtol': 1e-7, 'ftol': 1e-9,
                                    'maxfun': 5000})
            if res.fun < best_side * 1.02:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
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

    # 2c: Tight coordinate descent
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5.0:
        tight_time = min(remaining * 0.6, 18.0)
        t_side, t_pos, t_ang = polish_cd(best_positions, best_angles, tight_time, step=0.01, min_step=0.0005)
        if t_side < best_side:
            best_side = t_side
            best_positions = t_pos
            best_angles = t_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
