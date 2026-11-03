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
        
        # Vectorized coarse scan
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        best_per_angle = np.max(side_vals, axis=1)
        best_idx = np.argmin(best_per_angle)
        best_side = best_per_angle[best_idx]
        best_phi = float(best_idx)
        
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

    # Multiple initial configurations
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    configs = [
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
         (1, -1), (-1, 1), (2, -1), (-2, 1),
         (1, 1), (-1, -1), (2, 0)],
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
         (1, 1), (-1, -1), (2, -1), (-2, 1),
         (1, -1), (-1, 1), (2, 0)],
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
         (1, -1), (-1, 1), (2, -1), (-2, 1),
         (1, 1), (-1, -1), (0, 2)],
    ]
    
    best_init_side = float('inf')
    best_init_pos = None
    best_init_ang = None
    
    for cfg in configs:
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(cfg):
            pos[idx] = i * a1 + j * a2
            ang[idx] = 0.0
        side, _, _, _ = compute_min_outer_side(pos, ang)
        if side < best_init_side:
            best_init_side = side
            best_init_pos = pos.copy()
            best_init_ang = ang.copy()
    
    base_positions = best_init_pos
    base_angles = best_init_ang

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Quick local optimization
    quick_cd_time = min(8.0, DEADLINE * 0.06)
    t_cd_start = time.time()
    step_cd = 0.05
    min_step_cd = 0.001
    current_side_init, _, _, _ = compute_min_outer_side(base_positions, base_angles)
    
    while time.time() - t_cd_start < quick_cd_time and step_cd > min_step_cd:
        improved = False
        for idx in range(N):
            for dim in range(3):
                for direction in [1, -1]:
                    new_pos = base_positions.copy()
                    new_ang = base_angles.copy()
                    if dim == 0:
                        new_pos[idx, 0] += direction * step_cd
                    elif dim == 1:
                        new_pos[idx, 1] += direction * step_cd
                    else:
                        new_ang[idx] += direction * step_cd
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
                    if new_side < current_side_init - 1e-10:
                        base_positions = new_pos
                        base_angles = new_ang
                        current_side_init = new_side
                        improved = True
                        break
                if time.time() - t_cd_start > quick_cd_time:
                    break
            if time.time() - t_cd_start > quick_cd_time:
                break
        if not improved:
            step_cd *= 0.7
        if time.time() - t_cd_start > quick_cd_time:
            break
    
    best_side = current_side_init
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Main SA phase - 60% of budget
    sa_time_budget = DEADLINE * 0.60
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    T = 0.4
    T_min = 0.0002
    cooling = 0.99998
    step_size = 0.25
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 200000
    accept_count = 0
    total_count = 0
    
    def generate_move(current_pos, current_ang, step_sz):
        r = random.random()
        if r < 0.55:
            idxs = [random.randint(0, N - 1)]
        elif r < 0.82:
            idxs = random.sample(range(N), 2)
        elif r < 0.93:
            idxs = random.sample(range(N), 3)
        elif r < 0.98:
            idxs = random.sample(range(N), 4)
        else:
            idxs = list(range(N))
            
        new_pos = current_pos.copy()
        new_ang = current_ang.copy()
        
        eff_step = step_sz
        if len(idxs) == N:
            eff_step = step_sz * 0.2
        elif len(idxs) >= 3:
            eff_step = step_sz * 0.55
            
        for idx in idxs:
            dx = random.gauss(0, eff_step)
            dy = random.gauss(0, eff_step)
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

    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        new_positions, new_angles, moved_idxs = generate_move(current_positions, current_angles, step_size)
        
        valid = check_validity(new_positions, new_angles, moved_idxs)
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.004: step_size = 0.004
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
            if T > T_min and random.random() < math.exp(-delta / T):
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
        
        if total_count % 250 == 0:
            acc_rate = accept_count / max(1, total_count)
            if acc_rate < 0.06:
                step_size *= 0.7
                if step_size < 0.004: step_size = 0.004
            elif acc_rate > 0.35:
                step_size *= 1.3
                if step_size > 0.9: step_size = 0.9
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.004: step_size = 0.004

    # Polish Phase - remaining time
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 5.0:
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
                    step *= 0.7
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.5, 25.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 10.0:
            def polish_objective(params):
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = params[3*i]
                    pos[i, 1] = params[3*i+1]
                    ang[i] = params[3*i+2]
                    
                angles_rad = np.radians(ang)
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                        h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            d = np.linalg.norm(pos[i] - pos[j])
                            penalty += max(0, 2.0 - d) ** 2 * 5000
                if penalty > 0:
                    return 1e6 + penalty
                    
                side, phi, cx, cy = compute_min_outer_side(pos, ang)
                return side

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-12})
                
                if res.fun < best_side and res.fun < 1e5:
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
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            best_positions[idx] = i * a1 + j * a2
            best_angles[idx] = 0.0
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
