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

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 12.0:
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
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        all_verts = np.stack((verts_x, verts_y), axis=-1)
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
            return float(np.max(side_vals))

        # Coarse scan: 2 degree steps
        phi_vals = np.arange(0, 60, 2.0)
        phi_rad = np.radians(phi_vals)
        normal_angles = phi_rad[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)
        side_vals = (2.0 / sqrt3) * max_projs
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_side = float(np.max(side_vals[best_idx]))
        best_phi = float(phi_vals[best_idx])

        # Fine tune with bounded scalar minimization
        lo = max(0.0, best_phi - 2.0)
        hi = min(60.0, best_phi + 2.0)
        try:
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_side:
                best_side = float(res.fun)
                best_phi = float(res.x)
        except Exception:
            pass

        return best_side, best_phi, float(center[0]), float(center[1])

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    def make_honeycomb():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    def make_ring_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            a = k * np.pi / 3
            positions[k, 0] = 2*sqrt3 * math.cos(a)
            positions[k, 1] = 2*sqrt3 * math.sin(a)
        positions[6] = [0, 0]
        for k in range(5):
            a = k * 2*np.pi/5 + np.pi/5
            positions[7+k, 0] = 3.5*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3.5*sqrt3 * math.sin(a)
        return positions, angles

    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
            positions[k+6, 0] = k * 2*sqrt3 + sqrt3/2
            positions[k+6, 1] = 3.0
        return positions, angles

    # Evaluate initial configurations
    inits = [make_honeycomb(), make_ring_config(), make_two_rows()]
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos_init, ang_init in inits:
        if is_valid_configuration(pos_init, ang_init):
            s, phi, cx, cy = compute_min_outer_side(pos_init, ang_init)
            if s < best_side:
                best_side = s
                best_positions = pos_init.copy()
                best_angles = ang_init.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Phase 1: Simulated Annealing
    sa_time_budget = DEADLINE * 0.55
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side = best_side
    current_phi = best_phi
    current_cx = best_cx
    current_cy = best_cy

    T = 0.4
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.25
    step_decay = 0.99998

    step_count = 0
    max_steps = 300000
    accept_count = 0
    total_count = 0

    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
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
            if step_size < 0.005: step_size = 0.005
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

        if total_count % 500 == 0:
            acc_rate = accept_count / max(total_count, 1)
            if acc_rate < 0.1:
                step_size *= 0.85
                if step_size < 0.005: step_size = 0.005
            elif acc_rate > 0.5:
                step_size *= 1.15
                if step_size > 0.8: step_size = 0.8
            accept_count = 0
            total_count = 0

        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005

    # Phase 2: Second SA restart with perturbation
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 20.0:
        restart_budget = min(remaining_time * 0.25, 25.0)
        restart_pos = best_positions.copy()
        restart_ang = best_angles.copy()
        for i in range(N):
            restart_pos[i, 0] += random.gauss(0, 0.4)
            restart_pos[i, 1] += random.gauss(0, 0.4)
            restart_ang[i] += random.gauss(0, 8.0)

        if is_valid_configuration(restart_pos, restart_ang):
            r_side, r_phi, r_cx, r_cy = compute_min_outer_side(restart_pos, restart_ang)
            if r_side < best_side:
                best_side = r_side
                best_positions = restart_pos
                best_angles = restart_ang
                best_phi = r_phi
                best_cx = r_cx
                best_cy = r_cy

            # Quick SA from restart
            rp = restart_pos.copy()
            ra = restart_ang.copy()
            rs = r_side
            rT = 0.15
            rstep = 0.15
            rstart = time.time()
            rmax_steps = 50000
            rstep_count = 0
            racc = 0
            rtot = 0
            while time.time() - rstart < restart_budget and rstep_count < rmax_steps:
                rstep_count += 1
                rtot += 1
                ridx = random.randint(0, N-1)
                np_ = rp.copy()
                na = ra.copy()
                np_[ridx, 0] += random.gauss(0, rstep)
                np_[ridx, 1] += random.gauss(0, rstep)
                na[ridx] += random.gauss(0, 3.0)
                ar = np.radians(na)
                valid_r = True
                hm = (np_[ridx,0], np_[ridx,1], ar[ridx])
                for j in range(N):
                    if j == ridx: continue
                    ho = (np_[j,0], np_[j,1], ar[j])
                    if not hexagons_disjoint_fast(hm, ho):
                        valid_r = False
                        break
                if not valid_r:
                    rT *= 0.9999
                    if rT < 0.0001: rT = 0.0001
                    rstep *= 0.99998
                    if rstep < 0.005: rstep = 0.005
                    continue
                ns, nphi, ncx, ncy = compute_min_outer_side(np_, na)
                if ns < rs:
                    rp = np_; ra = na; rs = ns
                    racc += 1
                    if ns < best_side:
                        best_side = ns
                        best_positions = np_.copy()
                        best_angles = na.copy()
                        best_phi = nphi
                        best_cx = ncx
                        best_cy = ncy
                else:
                    d = ns - rs
                    if d < 0: d = 0
                    if random.random() < math.exp(-d / rT):
                        rp = np_; ra = na; rs = ns
                        racc += 1
                        if ns < best_side:
                            best_side = ns
                            best_positions = np_.copy()
                            best_angles = na.copy()
                            best_phi = nphi
                            best_cx = ncx
                            best_cy = ncy
                if rtot % 300 == 0:
                    arate = racc / max(rtot, 1)
                    if arate < 0.1:
                        rstep *= 0.85
                        if rstep < 0.005: rstep = 0.005
                    elif arate > 0.5:
                        rstep *= 1.15
                        if rstep > 0.5: rstep = 0.5
                    racc = 0; rtot = 0
                rT *= 0.9999
                if rT < 0.0001: rT = 0.0001
                rstep *= 0.99998
                if rstep < 0.005: rstep = 0.005

    # Phase 3: Coordinate Descent Polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        cd_time = min(remaining_time * 0.4, 20.0)
        cd_pos = best_positions.copy()
        cd_ang = best_angles.copy()
        cd_side, _, _, _ = compute_min_outer_side(cd_pos, cd_ang)
        cd_best_side = cd_side
        cd_best_pos = cd_pos.copy()
        cd_best_ang = cd_ang.copy()

        t_cd_start = time.time()
        cd_step = 0.03
        cd_min_step = 0.0005

        while time.time() - t_cd_start < cd_time and cd_step > cd_min_step:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        np_ = cd_best_pos.copy()
                        na = cd_best_ang.copy()
                        if dim == 0:
                            np_[idx, 0] += direction * cd_step
                        elif dim == 1:
                            np_[idx, 1] += direction * cd_step
                        else:
                            na[idx] += direction * cd_step

                        ar = np.radians(na)
                        valid_cd = True
                        hm = (np_[idx,0], np_[idx,1], ar[idx])
                        for j in range(N):
                            if j == idx: continue
                            ho = (np_[j,0], np_[j,1], ar[j])
                            if not hexagons_disjoint_fast(hm, ho):
                                valid_cd = False
                                break
                        if not valid_cd:
                            continue

                        ns, _, _, _ = compute_min_outer_side(np_, na)
                        if ns < cd_best_side - 1e-12:
                            cd_best_pos = np_
                            cd_best_ang = na
                            cd_best_side = ns
                            improved = True
                            break

                    if time.time() - t_cd_start > cd_time:
                        break
                else:
                    continue
                break
            else:
                cd_step *= 0.7
                continue
            break

        if cd_best_side < best_side:
            best_side = cd_best_side
            best_positions = cd_best_pos
            best_angles = cd_best_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Phase 4: Nelder-Mead Final Polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 8.0:
        def nm_objective(params):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            angles_rad = np.radians(ang)
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos[i,0], pos[i,1], angles_rad[i])
                    h2 = (pos[j,0], pos[j,1], angles_rad[j])
                    if not hexagons_disjoint_fast(h1, h2):
                        return 1e6
            side, _, _, _ = compute_min_outer_side(pos, ang)
            return side

        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]

        try:
            res = minimize(nm_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'initial_simplex': 0.01})
            if res.fun < best_side:
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

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos_f, ang_f in inits:
            if is_valid_configuration(pos_f, ang_f):
                best_positions = pos_f.copy()
                best_angles = ang_f.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
