import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

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
        """Fast side length estimate without minimize_scalar refinement."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        # Vectorized coarse scan: evaluate all 60 angles at once
        phi_vals = np.radians(np.arange(0, 60, 1))  # (60,)
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3  # (60, 6)
        nx = np.cos(normal_angles)  # (60, 6)
        ny = np.sin(normal_angles)  # (60, 6)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)  # (72, 60, 6)
        max_projs = np.max(abs_proj, axis=0)  # (60, 6)
        side_vals = (2.0 / sqrt3) * max_projs  # (60, 6)
        side_per_phi = np.max(side_vals, axis=1)  # (60,)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        return best_side, best_phi, center[0], center[1]

    def compute_min_outer_side(positions, angles_deg):
        """Full side length with minimize_scalar refinement."""
        best_side, best_phi, cx, cy = fast_side_length(positions, angles_deg)
        
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        # Fine tune with minimize_scalar
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

        return best_side, best_phi, cx, cy

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
        """Tighter hexagonal arrangement"""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Center
        pos[0] = [0, 0]
        # First ring (6 hexes)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        # Second ring partial (5 more to make 12)
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_tight_cluster():
        """Very tight cluster: center + 6 around + 5 more tightly packed"""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Center
        pos[0] = [0, 0]
        # First ring (6 hexes) at distance 2*sqrt(3)
        d1 = 2 * sqrt3
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [d1 * math.cos(theta), d1 * math.sin(theta)]
        # Additional 5 hexes in gaps, slightly closer than 4*sqrt(3)
        d2 = 3.5 * sqrt3
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [d2 * math.cos(theta), d2 * math.sin(theta)]
        return pos, ang

    init_configs = [make_honeycomb(), make_ring_config(), make_two_rows(), make_compact_hex(), make_tight_cluster()]

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

    # Strategy: Run SA for 50% of the time, then polish for the rest
    sa_time_budget = DEADLINE * 0.50

    # SA run with adaptive parameters
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)

    T = 0.3
    T_min = 0.0001
    cooling = 0.9999
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
            if step_size < 0.003: step_size = 0.003
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
        if total_count % 300 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.08:
                step_size *= 0.8
                if step_size < 0.003: step_size = 0.003
            elif acc_rate > 0.4:
                step_size *= 1.15
                if step_size > 0.8: step_size = 0.8
            accept_count = 0
            total_count = 0

        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # 1. Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.04
            min_step = 0.0005
            
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
                    step *= 0.7
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.25, 8.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 2. L-BFGS-B on smooth surrogate with soft collision penalty
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 15.0:
            # Precompute all vertex pairs (i, j) for i < j, N=12 -> 66 pairs
            _pair_indices = [(i, j) for i in range(N) for j in range(i+1, N)]
            _n_pairs = len(_pair_indices)

            def smooth_outer_side(params):
                """Smooth objective: outer side length + log-sum-exp soft collision penalty."""
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = params[3*i]
                    pos[i, 1] = params[3*i+1]
                    ang[i] = params[3*i+2]
                
                angles_rad = np.radians(ang)
                
                # Compute outer side length (smooth version using all 60 scan angles)
                all_verts = get_all_vertices(pos, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                offsets = verts_flat - center
                
                # Use a subset of angles for speed in the surrogate
                phi_vals = np.radians(np.arange(0, 60, 2))  # 30 angles
                normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
                max_projs = np.max(np.abs(proj), axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                side_per_phi = np.max(side_vals, axis=1)
                side = float(np.min(side_per_phi))
                
                # Log-sum-exp soft collision penalty:
                min_dist = 2.0 * sqrt3
                dx = np.array([pos[i, 0] - pos[j, 0] for i, j in _pair_indices])
                dy = np.array([pos[i, 1] - pos[j, 1] for i, j in _pair_indices])
                dists = np.sqrt(dx*dx + dy*dy)
                
                overlaps = np.maximum(0.0, min_dist - dists)
                beta = 50.0
                scaled = beta * overlaps
                max_s = np.max(scaled)
                if max_s > 500:
                    lse = max_s / beta + np.log(np.sum(np.exp(scaled - max_s))) / beta
                else:
                    lse = np.log(np.sum(np.exp(scaled))) / beta
                penalty = 10.0 * lse
                
                return side + penalty
            
            def smooth_grad(params):
                """Numerical gradient for the smooth objective."""
                eps = 1e-5
                grad = np.zeros_like(params)
                f0 = smooth_outer_side(params)
                for k in range(len(params)):
                    params_p = params.copy()
                    params_p[k] += eps
                    f_p = smooth_outer_side(params_p)
                    params[k] = params_p[k]
                    grad[k] = (f_p - f0) / eps
                    params[k] = params_p[k]
                return grad
            
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                # Use L-BFGS-B with numerical gradient
                res = minimize(smooth_outer_side, init_params, method='L-BFGS-B',
                               jac=smooth_grad,
                               options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000})
                
                if res.fun < best_side + 1.0 and res.fun < 100:
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
        
        # 3. Second round of coordinate descent with smaller steps
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
            cd_time2 = min(remaining_time * 0.7, 14.0)
            cd_side2, cd_pos2, cd_ang2 = polish_coordinate_descent(best_positions, best_angles, cd_time2)
            if cd_side2 < best_side:
                best_side = cd_side2
                best_positions = cd_pos2
                best_angles = cd_ang2
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        # Fall back to best valid initial config
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Final tight CD polish if time remains
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 1.0:
        def polish_cd_tight(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_t = current_side
            best_pos_t = pos.copy()
            best_ang_t = ang.copy()
            
            t_start = time.time()
            step = 0.008
            min_step = 0.0001
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_t.copy()
                            new_ang = best_ang_t.copy()
                            
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
                            
                            if new_side < best_side_t - 1e-12:
                                best_pos_t = new_pos
                                best_ang_t = new_ang
                                best_side_t = new_side
                                improved = True
                                break
                            
                        if time.time() - t_start > time_limit:
                            return best_side_t, best_pos_t, best_ang_t
                    else:
                        continue
                    break
                else:
                    step *= 0.6
                    continue
                break
            
            return best_side_t, best_pos_t, best_ang_t
        
        tight_time = min(remaining_time * 0.9, 8.0)
        t_side, t_pos, t_ang = polish_cd_tight(best_positions, best_angles, tight_time)
        if t_side < best_side:
            best_side = t_side
            best_positions = t_pos
            best_angles = t_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
