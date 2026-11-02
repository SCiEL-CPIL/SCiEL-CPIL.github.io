import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 140.0  # Safety margin before 150s wall-clock kill

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
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
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
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        en = edge_normals
        en1x = en[:, 0] * c1 - en[:, 1] * s1
        en1y = en[:, 0] * s1 + en[:, 1] * c1
        en2x = en[:, 0] * c2 - en[:, 1] * s2
        en2y = en[:, 0] * s2 + en[:, 1] * c2
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
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
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
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Honeycomb initial configuration
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    initial_positions = np.zeros((N, 2))
    initial_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        initial_positions[idx] = i * a1 + j * a2
        initial_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = initial_positions.copy()
    best_angles = initial_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    current_positions = initial_positions.copy()
    current_angles = initial_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    best_side = current_side
    best_phi = current_phi
    best_cx = current_cx
    best_cy = current_cy

    T = 0.3
    T_min = 0.0001
    cooling = 0.9998
    step_size = 0.2
    step_decay = 0.99995
    step_count = 0
    max_steps = 120000

    while time.time() - t0 < DEADLINE * 0.75 and step_count < max_steps:
        step_count += 1
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
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
        T *= cooling
        if T < T_min:
            T = T_min
        step_size *= step_decay
        if step_size < 0.005:
            step_size = 0.005

    # Polish with Nelder-Mead
    def compute_overlap_penalty(pos, angles_rad, soft=50.0):
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    dx = h1[0] - h2[0]
                    dy = h1[1] - h2[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    overlap_depth = max(0.0, 2.0 - dist)
                    penalty += soft * overlap_depth * overlap_depth
        return penalty

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
                    return 1e4 + compute_overlap_penalty(pos, angles_rad, soft=100.0)
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        return side

    def polish_objective_smooth(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        angles_rad = np.radians(ang)
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        penalty = compute_overlap_penalty(pos, angles_rad, soft=50.0)
        return side + penalty

    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = best_positions[i, 0]
        init_params[3*i+1] = best_positions[i, 1]
        init_params[3*i+2] = best_angles[i]

    polish_time_budget = DEADLINE - (time.time() - t0)
    if polish_time_budget > 5.0:
        try:
            res = minimize(polish_objective_smooth, init_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            new_pos = np.zeros((N, 2))
            new_ang = np.zeros(N)
            for i in range(N):
                new_pos[i, 0] = res.x[3*i]
                new_pos[i, 1] = res.x[3*i+1]
                new_ang[i] = res.x[3*i+2]
            if is_valid_configuration(new_pos, new_ang):
                cand_side, cand_phi, cand_cx, cand_cy = compute_min_outer_side(new_pos, new_ang)
                if cand_side < best_side:
                    best_side = cand_side
                    best_positions = new_pos
                    best_angles = new_ang
                    best_phi = cand_phi
                    best_cx = cand_cx
                    best_cy = cand_cy
                    if time.time() - t0 < DEADLINE - 3.0:
                        init_params2 = np.zeros(3*N)
                        for i in range(N):
                            init_params2[3*i] = best_positions[i, 0]
                            init_params2[3*i+1] = best_positions[i, 1]
                            init_params2[3*i+2] = best_angles[i]
                        res2 = minimize(polish_objective, init_params2, method='Nelder-Mead',
                                       options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
                        if res2.fun < 1e3:
                            new_pos2 = np.zeros((N, 2))
                            new_ang2 = np.zeros(N)
                            for i in range(N):
                                new_pos2[i, 0] = res2.x[3*i]
                                new_pos2[i, 1] = res2.x[3*i+1]
                                new_ang2[i] = res2.x[3*i+2]
                            if is_valid_configuration(new_pos2, new_ang2):
                                cand_side2, cand_phi2, cand_cx2, cand_cy2 = compute_min_outer_side(new_pos2, new_ang2)
                                if cand_side2 < best_side:
                                    best_side = cand_side2
                                    best_positions = new_pos2
                                    best_angles = new_ang2
                                    best_phi = cand_phi2
                                    best_cx = cand_cx2
                                    best_cy = cand_cy2
        except Exception:
            pass

    final_valid = is_valid_configuration(best_positions, best_angles)
    if not final_valid:
        best_positions = initial_positions.copy()
        best_angles = initial_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
