import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 143.0  # Safety margin before 150s wall-clock kill

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

    def hexagon_separation(h1, h2):
        """Return positive separation distance between two hexagons (0 if touching/overlapping)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return math.sqrt(dist_sq) - 2.0  # definitely disjoint, approx
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
        # Check all separating axes
        min_sep = 0.0
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            sep = min(p2.max() - p1.max(), p1.max() - p2.min())
            # Actually: separation along this axis
            # If p1.max() < p2.min(): sep = p2.min() - p1.max()
            # If p2.max() < p1.min(): sep = p1.min() - p2.max()
            # Else: overlapping on this axis, sep = 0 for this axis
            sep1 = p2.min() - p1.max()
            sep2 = p1.min() - p2.max()
            if sep1 > 0 or sep2 > 0:
                min_sep = max(min_sep, max(sep1, sep2))
            else:
                # Overlapping on this axis - the hexagons might still be separated by another axis
                # But for separation distance, we need the minimum over all axes where they don't overlap
                pass
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            sep1 = p2.min() - p1.max()
            sep2 = p1.min() - p2.max()
            if sep1 > 0 or sep2 > 0:
                min_sep = max(min_sep, max(sep1, sep2))
            else:
                pass
        return min_sep

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

        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()

        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
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

    def min_separation(positions, angles_rad):
        """Compute minimum separation between all pairs of hexagons."""
        min_sep = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                sep = hexagon_separation(h1, h2)
                if sep < min_sep:
                    min_sep = sep
        return min_sep

    def objective(params):
        """Objective: outer side length + penalty for overlaps."""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        angles_rad = np.radians(ang)
        
        # Compute separation
        min_sep = min_separation(pos, angles_rad)
        if min_sep < 0:
            penalty = 10.0 * abs(min_sep)
        else:
            penalty = 0.0
        
        # Compute outer side
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_vals = np.radians(np.arange(0, 60, 2))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side = np.max(side_vals)
        
        return side + penalty

    def objective_smooth(params, beta=50.0):
        """Smooth objective for L-BFGS-B."""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        angles_rad = np.radians(ang)
        
        # Compute separations
        seps = []
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                sep = hexagon_separation(h1, h2)
                seps.append(sep)
        
        # Soft penalty: smooth min of separations
        if len(seps) > 0:
            min_sep = min(seps)
            if min_sep < 0:
                penalty = 10.0 * abs(min_sep)
            else:
                # Encourage positive separation
                penalty = 0.0
        else:
            penalty = 0.0
        
        # Compute outer side
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_vals = np.radians(np.arange(0, 60, 2))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side = np.max(side_vals)
        
        return side + penalty

    # Generate initial configurations
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])

    def make_honeycomb():
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        return pos, ang

    def make_ring_config():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 6] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_two_rows():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i] = [i * 2 * sqrt3, 0]
            pos[i + 6] = [i * 2 * sqrt3 + sqrt3, 1.5 * 2]
        return pos, ang

    def make_compact_hex():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_random_config(seed):
        rng = np.random.RandomState(seed)
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Start from honeycomb and perturb
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        global_rot = rng.uniform(0, 2 * np.pi)
        c_g, s_g = math.cos(global_rot), math.sin(global_rot)
        for idx, (i, j) in enumerate(lattice_coords):
            x, y = i * a1[0] + j * a2[0], i * a1[1] + j * a2[1]
            rx = x * c_g - y * s_g
            ry = x * s_g + y * c_g
            rx += rng.uniform(-0.5, 0.5)
            ry += rng.uniform(-0.5, 0.5)
            pos[idx] = [rx, ry]
            ang[idx] = rng.uniform(0, 60)
        return pos, ang

    init_configs = [make_honeycomb(), make_ring_config(), make_two_rows(), make_compact_hex()]
    for s in range(8):
        init_configs.append(make_random_config(s))

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Evaluate all initial configs
    for pos0, ang0 in init_configs:
        if is_valid_configuration(pos0, ang0):
            s, phi, cx, cy = compute_min_outer_side(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Multi-start Nelder-Mead + L-BFGS-B optimization
    elapsed = time.time() - t0
    time_per_start = (DEADLINE - 10.0 - elapsed) / 12  # 12 optimization starts

    for start_idx in range(12):
        if time.time() - t0 > DEADLINE - 15.0:
            break

        # Choose initial config
        if start_idx < len(init_configs):
            pos0, ang0 = init_configs[start_idx]
        else:
            pos0, ang0 = make_random_config(start_idx + 100)

        if not is_valid_configuration(pos0, ang0):
            continue

        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = pos0[i, 0]
            init_params[3*i+1] = pos0[i, 1]
            init_params[3*i+2] = ang0[i]

        # Nelder-Mead
        t_start = time.time()
        try:
            res_nm = minimize(objective, init_params, method='Nelder-Mead',
                              options={'maxiter': 2000, 'fatol': 1e-10, 'xatol': 1e-8,
                                       'adaptive': True})
            if res_nm.fun < best_side + 0.5:
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

        # L-BFGS-B from best found
        if time.time() - t0 > DEADLINE - 12.0:
            break

        current_params = np.zeros(3*N)
        for i in range(N):
            current_params[3*i] = best_positions[i, 0]
            current_params[3*i+1] = best_positions[i, 1]
            current_params[3*i+2] = best_angles[i]

        try:
            res_lbfgs = minimize(objective, current_params, method='L-BFGS-B',
                                 options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res_lbfgs.fun < best_side + 0.5:
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
        except Exception:
            pass

    # Final polish: coordinate descent
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        def polish_cd(pos, ang, time_limit):
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
                            if new_side < best_side_cd - 1e-12:
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
                    step *= 0.5
                    continue
                break
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.6, 10.0)
        cd_side, cd_pos, cd_ang = polish_cd(best_positions, best_angles, cd_time)
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final phi refinement
    try:
        angles_rad_final = np.radians(best_angles)
        all_verts_final = get_all_vertices(best_positions, angles_rad_final)
        verts_flat_final = all_verts_final.reshape(-1, 2)
        center_final = np.mean(verts_flat_final, axis=0)
        offsets_final = verts_flat_final - center_final

        def exact_side_for_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets_final[:, 0:1] * nxx[None, :] + offsets_final[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()

        phi_grid = np.linspace(max(0, best_phi - 1.0), min(60, best_phi + 1.0), 201)
        sides_grid = np.array([exact_side_for_phi(p) for p in phi_grid])
        best_idx = np.argmin(sides_grid)
        if sides_grid[best_idx] < best_side - 1e-12:
            best_phi = float(phi_grid[best_idx])
            best_side = float(sides_grid[best_idx])
            best_cx = float(center_final[0])
            best_cy = float(center_final[1])
    except Exception:
        pass

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
