import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 148.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
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
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
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

    def fast_side_length(positions, angles_deg):
        """Fast side-length estimate using coarse scan only (no minimize_scalar)."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)
        side_vals = (2.0 / sqrt3) * max_projs
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_side = np.max(side_vals[best_idx])
        best_phi = float(best_idx) * 0.5
        return best_side, best_phi, center[0], center[1]

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

        # Vectorized coarse scan: evaluate all 120 angles at once (0.5° steps)
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)
        side_vals = (2.0 / sqrt3) * max_projs
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_side = np.max(side_vals[best_idx])
        best_phi = float(best_idx) * 0.5

        # Fine tune with minimize_scalar
        try:
            lo = max(0.0, best_phi - 0.5)
            hi = min(60.0, best_phi + 0.5)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def compute_min_outer_side_smooth(positions, angles_deg, beta=50.0):
        """Smooth surrogate using log-sum-exp for the outer max."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        def eval_phi_smooth(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            # log-sum-exp for smooth max with scaling by beta
            scaled = abs_proj * beta
            m = np.max(scaled, axis=0)
            log_sum_exp = (m + np.log(np.sum(np.exp(scaled - m[None, :]), axis=0))) / beta
            side_vals = (2.0 / sqrt3) * log_sum_exp
            return np.max(side_vals)

        # Coarse scan
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)
        scaled = abs_proj * beta
        m = np.max(scaled, axis=1)
        log_sum_exp = (m + np.log(np.sum(np.exp(scaled - m[:, None, :]), axis=1))) / beta
        side_vals = (2.0 / sqrt3) * log_sum_exp
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_side = np.max(side_vals[best_idx])
        best_phi = float(best_idx) * 0.5

        # Fine tune
        try:
            lo = max(0.0, best_phi - 0.5)
            hi = min(60.0, best_phi + 0.5)
            res = minimize_scalar(eval_phi_smooth, bounds=(lo, hi), method='bounded',
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
        # 7 hexagons in a ring around center, 5 more
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        # Ring of 6 at distance ~2*sqrt(3)
        for k in range(6):
            a = k * np.pi / 3
            positions[k, 0] = 2*sqrt3 * math.cos(a)
            positions[k, 1] = 2*sqrt3 * math.sin(a)
        # Center
        positions[6] = [0, 0]
        # 5 more outside
        for k in range(5):
            a = k * 2*np.pi/5 + np.pi/5
            positions[7+k, 0] = 3.5*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3.5*sqrt3 * math.sin(a)
        return positions, angles

    def make_two_rows():
        # Two rows of 6 hexagons
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
            positions[k+6, 0] = k * 2*sqrt3 + sqrt3/2
            positions[k+6, 1] = 3.0
        return positions, angles

    base_positions, base_angles = make_honeycomb()

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Evaluate multiple initial configurations
    inits = [make_honeycomb(), make_ring_config(), make_two_rows()]
    for pos_init, ang_init in inits:
        s, phi, cx, cy = compute_min_outer_side(pos_init, ang_init)
        if s < best_side:
            best_side = s
            best_positions = pos_init.copy()
            best_angles = ang_init.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    # Strategy: Run SA for 50% of the time (using fast_side_length), then Polish for the rest
    sa_time_budget = DEADLINE * 0.50

    # Start SA from best initial configuration
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, _, _, _ = fast_side_length(current_positions, current_angles)
    current_phi = best_phi
    current_cx = best_cx
    current_cy = best_cy

    T = 0.3
    T_min = 0.00005
    cooling = 0.99995
    step_size = 0.2
    step_decay = 0.99999

    step_count = 0
    max_steps = 200000
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
        da = random.gauss(0, 5.0)

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

        new_side, new_phi, new_cx, new_cy = fast_side_length(new_positions, new_angles)

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

        # Adaptive step size
        if total_count % 500 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.1:
                step_size *= 0.8
                if step_size < 0.005: step_size = 0.005
            elif acc_rate > 0.5:
                step_size *= 1.2
                if step_size > 1.0: step_size = 1.0
            accept_count = 0
            total_count = 0

        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)

    if remaining_time > 2.0:
        # Quick SA restart from a perturbed version if time allows
        if remaining_time > 30.0:
            restart_pos = best_positions.copy()
            restart_ang = best_angles.copy()
            for i in range(N):
                restart_pos[i, 0] += random.gauss(0, 0.5)
                restart_pos[i, 1] += random.gauss(0, 0.5)
                restart_ang[i] += random.gauss(0, 10.0)

            r_side, r_phi, r_cx, r_cy = compute_min_outer_side(restart_pos, restart_ang)
            if r_side < best_side:
                best_side = r_side
                best_positions = restart_pos
                best_angles = restart_ang
                best_phi = r_phi
                best_cx = r_cx
                best_cy = r_cy
            remaining_time = DEADLINE - (time.time() - t0)

        # 1. Smooth surrogate polish using log-sum-exp with L-BFGS
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

            side, phi, cx, cy = compute_min_outer_side_smooth(pos, ang, beta=beta)
            return side

        smooth_time = min(remaining_time * 0.4, 30.0)
        if smooth_time > 3.0:
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]

            try:
                # Try L-BFGS first (faster convergence for smooth objectives)
                res_lbfgs = minimize(smooth_objective, init_params, method='L-BFGS-B',
                                     options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8,
                                              'maxls': 50})
                
                if res_lbfgs.fun < best_side + 0.3:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res_lbfgs.x[3*i]
                        new_pos[i, 1] = res_lbfgs.x[3*i+1]
                        new_ang[i] = res_lbfgs.x[3*i+2]

                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
                
                # Then Nelder-Mead for local refinement
                remaining_smooth = DEADLINE - (time.time() - t0) - 5.0
                if remaining_smooth > 5.0:
                    init_params2 = np.zeros(3*N)
                    for i in range(N):
                        init_params2[3*i] = best_positions[i, 0]
                        init_params2[3*i+1] = best_positions[i, 1]
                        init_params2[3*i+2] = best_angles[i]

                    res_nm = minimize(smooth_objective, init_params2, method='Nelder-Mead',
                                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                               'maxfev': 8000})

                    if res_nm.fun < best_side + 0.2:
                        new_pos = np.zeros((N, 2))
                        new_ang = np.zeros(N)
                        for i in range(N):
                            new_pos[i, 0] = res_nm.x[3*i]
                            new_pos[i, 1] = res_nm.x[3*i+1]
                            new_ang[i] = res_nm.x[3*i+2]

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

        # 2. Coordinate Descent Polish (high precision) with golden section search
        remaining_time = DEADLINE - (time.time() - t0)

        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()

            t_start = time.time()
            step = 0.02
            min_step = 0.0001

            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        # Golden section search in both directions
                        for direction in [1, -1]:
                            # Try to find improvement via golden section
                            a = 0.0
                            b = direction * step * 5
                            
                            # Evaluate at golden ratio points
                            gr = (1 + math.sqrt(5)) / 2
                            c = b - (b - a) / gr
                            d = a + (b - a) / gr
                            
                            # Get function values
                            def eval_at(offset_val):
                                new_pos = best_pos_cd.copy()
                                new_ang = best_ang_cd.copy()
                                if dim == 0:
                                    new_pos[idx, 0] += offset_val
                                elif dim == 1:
                                    new_pos[idx, 1] += offset_val
                                else:
                                    new_ang[idx] += offset_val
                                
                                angles_rad_new = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                for j in range(N):
                                    if j == idx: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        return float('inf')
                                
                                s, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                                return s
                            
                            fc = eval_at(c)
                            fd = eval_at(d)
                            
                            # Iterate golden section for a few steps
                            for _ in range(8):
                                if fc < fd:
                                    b = d
                                    d = c
                                    fd = fc
                                    c = b - (b - a) / gr
                                    fc = eval_at(c)
                                else:
                                    a = c
                                    c = d
                                    fc = fd
                                    d = a + (b - a) / gr
                                    fd = eval_at(d)
                            
                            best_offset = (a + b) / 2
                            best_val = min(fc, fd, eval_at(best_offset))
                            
                            if best_val < best_side_cd - 1e-12:
                                new_pos = best_pos_cd.copy()
                                new_ang = best_ang_cd.copy()
                                if dim == 0:
                                    new_pos[idx, 0] += best_offset
                                elif dim == 1:
                                    new_pos[idx, 1] += best_offset
                                else:
                                    new_ang[idx] += best_offset
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = best_val
                                improved = True
                                break

                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.6
                    continue
                break

            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.4, 25.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)

        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

        # 3. Final Nelder-Mead with exact objective + high-precision polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
            def polish_objective(params):
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

                side, phi, cx, cy = compute_min_outer_side(pos, ang)
                return side

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]

            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11,
                                        'maxfev': 6000})

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

        # 4. Final high-precision coordinate descent with binary search refinement
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            hp_step = 0.005
            hp_min_step = 0.00005
            t_hp = time.time()
            hp_side, _, _, _ = compute_min_outer_side(best_positions, best_angles)
            while time.time() - t_hp < remaining_time - 1.0 and hp_step > hp_min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            # Binary search for optimal offset in this direction
                            lo = 0.0
                            hi = direction * hp_step * 3
                            
                            def eval_offset(off):
                                new_pos = best_positions.copy()
                                new_ang = best_angles.copy()
                                if dim == 0:
                                    new_pos[idx, 0] += off
                                elif dim == 1:
                                    new_pos[idx, 1] += off
                                else:
                                    new_ang[idx] += off
                                angles_rad_new = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                for j in range(N):
                                    if j == idx: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        return float('inf')
                                s, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                                return s
                            
                            # Check if moving in this direction helps
                            mid = (lo + hi) / 2
                            f_mid = eval_offset(mid)
                            if f_mid >= hp_side:
                                continue
                            
                            # Binary search for the best offset
                            best_off = mid
                            best_f = f_mid
                            for _ in range(10):
                                mid = (lo + hi) / 2
                                f_mid = eval_offset(mid)
                                if f_mid < best_f:
                                    best_off = mid
                                    best_f = f_mid
                                # Determine which half to keep
                                q1 = lo + (hi - lo) * 0.25
                                q3 = lo + (hi - lo) * 0.75
                                f_q1 = eval_offset(q1)
                                f_q3 = eval_offset(q3)
                                if f_q1 < best_f:
                                    hi = q3
                                elif f_q3 < best_f:
                                    lo = q1
                                else:
                                    # Converged enough
                                    break
                            
                            if best_f < hp_side - 1e-14:
                                new_pos = best_positions.copy()
                                new_ang = best_angles.copy()
                                if dim == 0:
                                    new_pos[idx, 0] += best_off
                                elif dim == 1:
                                    new_pos[idx, 1] += best_off
                                else:
                                    new_ang[idx] += best_off
                                best_positions = new_pos
                                best_angles = new_ang
                                hp_side = best_f
                                improved = True
                                break
                        if time.time() - t_hp > remaining_time - 1.0:
                            break
                    else:
                        continue
                    break
                else:
                    hp_step *= 0.5
                    continue
                break
            if hp_side < best_side:
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        # Try each initial config as fallback
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
