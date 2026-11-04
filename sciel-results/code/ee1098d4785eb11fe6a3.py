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
    
    def compute_min_outer_side(positions, angles_deg, fine_tune=True):
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
        
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        max_sides = np.max(side_vals, axis=1)
        
        best_idx = np.argmin(max_sides)
        best_side = float(max_sides[best_idx])
        best_phi = float(best_idx)
        
        if fine_tune:
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
    
    def compute_smooth_outer_side_and_grad(positions, angles_deg, beta=50.0):
        """
        Compute smooth surrogate for outer side length and its gradient w.r.t. positions.
        Uses log-sum-exp to make min/max differentiable.
        Returns (smooth_side, grad_positions [N,2], grad_angles [N])
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Vectorized coarse scan to find good phi
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles_all = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx_all = np.cos(normal_angles_all)
        ny_all = np.sin(normal_angles_all)
        proj_all = offsets[:, 0:1, None] * nx_all[None, :, :] + offsets[:, 1:2, None] * ny_all[None, :, :]
        abs_proj_all = np.abs(proj_all)
        max_projs_all = np.max(abs_proj_all, axis=0)
        side_vals_all = (2.0 / sqrt3) * max_projs_all
        m_all = np.max(side_vals_all, axis=1)
        exp_all = np.exp(beta * (side_vals_all - m_all[:, None]))
        val_all = m_all + np.log(np.sum(exp_all, axis=1)) / beta
        
        best_idx = np.argmin(val_all)
        best_phi = float(phi_vals[best_idx])
        best_val = float(val_all[best_idx])
        
        # Fine tune phi
        lo = max(0.0, best_phi - 1.5)
        hi = min(60.0, best_phi + 1.5)
        try:
            def eval_smooth_phi(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                m = np.max(side_vals)
                return m + np.log(np.sum(np.exp(beta * (side_vals - m)))) / beta
            res = minimize_scalar(eval_smooth_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success:
                best_phi = res.x
                best_val = res.fun
        except Exception:
            pass
        
        # Compute gradients at best phi (vectorized)
        phi = np.radians(best_phi)
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        
        m = np.max(side_vals)
        exp_terms = np.exp(beta * (side_vals - m))
        sum_exp = np.sum(exp_terms)
        sm_weights = exp_terms / sum_exp
        
        # Per-vertex weights for each normal direction
        # For each normal k, compute softmax over |proj| for the max
        abs_proj_m = abs_proj - max_projs[None, :]
        exp_v = np.exp(beta * abs_proj_m)
        sum_v = np.sum(exp_v, axis=0)
        v_weights = exp_v / sum_v  # shape (72, 6)
        
        signs = np.sign(proj)  # shape (72, 6)
        
        # grad_pos[i, k] = (2/sqrt3) * sm_weights[k] * v_weights[i,k] * signs[i,k] * [nx[k], ny[k]]
        # We need to sum over k for each vertex i
        factor = (2.0 / sqrt3) * sm_weights  # shape (6,)
        weighted = factor[None, :] * v_weights * signs  # shape (72, 6)
        
        # grad_pos[i] = sum_k weighted[i,k] * [nx[k], ny[k]]
        grad_pos = np.zeros((72, 2))
        grad_pos[:, 0] = weighted @ nx
        grad_pos[:, 1] = weighted @ ny
        
        # Aggregate to hexagon centers
        grad_center = np.zeros((N, 2))
        for i in range(N):
            grad_center[i] = np.sum(grad_pos[i*6:(i+1)*6], axis=0)
        
        # Angle gradients: d(vert)/d(theta) = cross(vert_local, [1,0]) style
        # For vertex v of hex i: pos = center_i + R(theta_i) * local_v
        # d(pos)/d(theta_i) = R'(theta_i) * local_v = [[0,-sin],[cos,0]] * local_v
        grad_angle = np.zeros(N)
        for i in range(N):
            theta = angles_rad[i]
            c_t, s_t = math.cos(theta), math.sin(theta)
            # local vertices for hex i
            lx = _UNIT_HEX_VERTS[:, 0]
            ly = _UNIT_HEX_VERTS[:, 1]
            # d(vert_x)/dtheta = -sin(theta)*lx - cos(theta)*ly ... wait
            # R(theta) = [[cos, -sin],[sin, cos]]
            # R'(theta) = [[-sin, -cos],[cos, -sin]]
            # d(vert)/dtheta = R'(theta) * local = [-sin*lx - cos*ly, cos*lx - sin*ly]
            dv_dx = -s_t * lx - c_t * ly
            dv_dy = c_t * lx - s_t * ly
            # grad_angle[i] = sum_v grad_pos[i*6+v] . [dv_dx[v], dv_dy[v]]
            gp = grad_pos[i*6:(i+1)*6]
            grad_angle[i] = np.sum(gp[:, 0] * dv_dx + gp[:, 1] * dv_dy)
        
        return best_val, grad_center, grad_angle
    
    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True
    
    def make_grid_2x3(scale=1.0, rotation_deg=0.0, sx=1.0, sy=1.0, shift_x=0.0, shift_y=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        a1 = a1 * np.array([sx, 1.0])
        a2 = a2 * np.array([sx, sy])
        
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(2):
            for i in range(3):
                pos = i * a1 + j * a2
                positions[idx] = pos + np.array([shift_x, shift_y])
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles
    
    def make_grid_3x2(scale=1.0, rotation_deg=0.0, sx=1.0, sy=1.0, shift_x=0.0, shift_y=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        a1 = a1 * np.array([sx, 1.0])
        a2 = a2 * np.array([sx, sy])
        
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(3):
            for i in range(2):
                pos = i * a1 + j * a2
                positions[idx] = pos + np.array([shift_x, shift_y])
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles
    
    def make_hex_ring(center_pos, radius, rotation_deg=0.0):
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N + np.radians(rotation_deg)
            positions[i, 0] = center_pos[0] + radius * np.cos(theta)
            positions[i, 1] = center_pos[1] + radius * np.sin(theta)
            angles[i] = rotation_deg
        return positions, angles
    
    def make_two_clusters():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(2):
            for i in range(3):
                positions[idx] = i * a1 + j * a2
                angles[idx] = 0.0
                idx += 1
        offset = 3 * a1 + 0.5 * a2
        for j in range(2):
            for i in range(3):
                positions[idx] = (i) * a1 + j * a2 + offset
                angles[idx] = 0.0
                idx += 1
        return positions, angles
    
    # Generate initial configurations
    initial_configs = []
    
    for s in [0.95, 0.98, 1.0, 1.02, 1.05]:
        initial_configs.append(make_grid_2x3(scale=s))
        initial_configs.append(make_grid_3x2(scale=s))
    
    for rot in [5, 10, 15, 20, 30]:
        initial_configs.append(make_grid_2x3(scale=1.0, rotation_deg=rot))
        initial_configs.append(make_grid_3x2(scale=1.0, rotation_deg=rot))
    
    for sx in [0.95, 0.98, 1.0, 1.02, 1.05]:
        for sy in [0.95, 0.98, 1.0, 1.02, 1.05]:
            initial_configs.append(make_grid_2x3(scale=1.0, sx=sx, sy=sy))
    
    for r in [2.0, 2.5, 3.0, 3.5]:
        initial_configs.append(make_hex_ring([0, 0], r))
    
    initial_configs.append(make_two_clusters())
    
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    for scale in [0.9, 0.95, 1.0]:
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = (i * a1 + j * a2) * scale
            angles[idx] = 0.0
        initial_configs.append((positions, angles))
    
    # Add random perturbations of good configs for diversity
    random.seed(42)
    base_configs = [make_grid_2x3(1.0), make_grid_3x2(1.0), make_hex_ring([0,0], 2.5)]
    for base_pos, base_ang in base_configs:
        for _ in range(3):
            rp = base_pos + np.random.randn(N, 2) * 0.3
            ra = base_ang + np.random.randn(N) * 5.0
            initial_configs.append((rp, ra))
    
    # Staggered rows (offset every other row)
    for offset_scale in [0.5, 0.7, 1.0]:
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(2):
            for i in range(3):
                pos = i * a1 + j * a2 + np.array([offset_scale * sqrt3/2, 0]) * (j % 2)
                positions[idx] = pos
                angles[idx] = 0.0
                idx += 1
        initial_configs.append((positions, angles))
    
    # Two rows with slight rotation
    for rot in [3, 7, 12]:
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(2):
            for i in range(3):
                pos = i * a1 + j * a2
                positions[idx] = pos
                angles[idx] = rot
                idx += 1
        initial_configs.append((positions, angles))
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    for pos, ang in initial_configs:
        if time.time() - t0 > DEADLINE * 0.3:
            break
        try:
            side, phi, cx, cy = compute_min_outer_side(pos, ang, fine_tune=False)
            if is_valid_configuration(pos, ang) and side < best_side:
                best_side = side
                best_positions = pos.copy()
                best_angles = ang.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy
        except Exception:
            continue
    
    if best_positions is None:
        best_positions, best_angles = make_grid_2x3(1.0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Projected gradient method with smooth surrogate
    pg_deadline = t0 + DEADLINE * 0.45
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    lr = 0.05
    lr_decay = 0.999
    beta_schedule = [50.0, 30.0, 20.0, 15.0, 10.0]
    current_beta = beta_schedule[0]
    beta_phase = 0
    stagnant = 0
    
    while time.time() < pg_deadline:
        smooth_val, grad_pos, grad_ang = compute_smooth_outer_side_and_grad(
            current_positions, current_angles, beta=current_beta
        )
        
        new_positions = current_positions - lr * grad_pos
        new_angles = current_angles - lr * grad_ang * (180.0 / math.pi)
        
        angles_rad_new = np.radians(new_angles)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (new_positions[i, 0], new_positions[i, 1], angles_rad_new[i])
                h2 = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h1, h2):
                    overlap_penalty += 0.1
                    dx = new_positions[i, 0] - new_positions[j, 0]
                    dy = new_positions[i, 1] - new_positions[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        dx, dy = 0.1, 0.1
                        dist = math.sqrt(0.02)
                    push = 0.05 / dist
                    new_positions[i, 0] += dx * push
                    new_positions[i, 1] += dy * push
                    new_positions[j, 0] -= dx * push
                    new_positions[j, 1] -= dy * push
        
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=False)
        
        if new_side < current_side - 1e-10:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            stagnant = 0
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            stagnant += 1
        
        lr *= lr_decay
        if lr < 0.001:
            lr = 0.001
        
        if stagnant > 50 and beta_phase < len(beta_schedule) - 1:
            beta_phase += 1
            current_beta = beta_schedule[beta_phase]
            lr = max(lr, 0.02)
            stagnant = 0
    
    # SA phase with multi-restart
    sa_deadline = min(t0 + DEADLINE * 0.75, pg_deadline + DEADLINE * 0.3)
    
    # Run SA from multiple starting points
    sa_starts = [
        (best_positions.copy(), best_angles.copy()),
    ]
    # Add a few diverse starts
    for pos, ang in initial_configs[:5]:
        if is_valid_configuration(pos, ang):
            sa_starts.append((pos.copy(), ang.copy()))
    
    for start_idx, (sp, sa) in enumerate(sa_starts):
        if time.time() > sa_deadline:
            break
        
        remaining_sa = sa_deadline - time.time()
        if remaining_sa < 3.0:
            break
        # Allocate time proportionally
        time_alloc = remaining_sa / max(1, len(sa_starts) - start_idx)
        sa_phase_deadline = time.time() + time_alloc
        
        cur_p = sp.copy()
        cur_a = sa.copy()
        cur_side, cur_phi, cur_cx, cur_cy = compute_min_outer_side(cur_p, cur_a, fine_tune=False)
        
        T = 0.08
        T_min = 0.0001
        cooling = 0.99997
        step_size = 0.06
        step_decay = 0.99999
        
        while time.time() < sa_phase_deadline:
            idx = random.randint(0, N - 1)
            new_positions = cur_p.copy()
            new_angles = cur_a.copy()
            
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            da = random.gauss(0, 1.5)
            
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
            
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=False)
            
            if new_side < cur_side:
                cur_p = new_positions
                cur_a = new_angles
                cur_side = new_side
                cur_phi = new_phi
                cur_cx = new_cx
                cur_cy = new_cy
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
            else:
                delta = new_side - cur_side
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / T):
                    cur_p = new_positions
                    cur_a = new_angles
                    cur_side = new_side
                    cur_phi = new_phi
                    cur_cx = new_cx
                    cur_cy = new_cy
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_positions.copy()
                        best_angles = new_angles.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
            
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003
        
        # Update current for next phase
        if cur_side < current_side:
            current_positions = cur_p
            current_angles = cur_a
            current_side = cur_side
    
    # Coordinate descent polish
    remaining = DEADLINE - (time.time() - t0)
    
    if remaining > 5.0:
        cd_deadline = t0 + remaining * 0.45
        step = 0.03
        min_step = 0.00001
        
        while time.time() < cd_deadline and step > min_step:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()
                        
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
                        
                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                        
                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.4
            else:
                step = max(step * 0.75, min_step)
        
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # Nelder-Mead polish on full parameter space
        remaining2 = DEADLINE - (time.time() - t0)
        if remaining2 > 5.0:
            nm_deadline = t0 + remaining2 * 0.5
            
            def nm_objective(x):
                if time.time() > nm_deadline:
                    return 1e6
                pos = x[:N*2].reshape(N, 2)
                ang = x[N*2:]
                angles_rad = np.radians(ang)
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (pos[i,0], pos[i,1], angles_rad[i])
                        h2 = (pos[j,0], pos[j,1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return 1e6 + np.sum((pos - best_positions)**2) * 0.01
                side, _, _, _ = compute_min_outer_side(pos, ang, fine_tune=False)
                return side
            
            x0 = np.concatenate([best_positions.flatten(), best_angles])
            try:
                res = minimize(nm_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                       'adaptive': True})
                if res.fun < best_side - 1e-12 and time.time() < nm_deadline:
                    new_pos = res.x[:N*2].reshape(N, 2)
                    new_ang = res.x[N*2:]
                    if is_valid_configuration(new_pos, new_ang):
                        ns, np_, ncx, ncy = compute_min_outer_side(new_pos, new_ang)
                        if ns < best_side:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = ns
                            best_phi = np_
                            best_cx = ncx
                            best_cy = ncy
            except Exception:
                pass
        
        # Final gradient-based polish
        remaining3 = DEADLINE - (time.time() - t0)
        if remaining3 > 3.0:
            pg2_deadline = t0 + remaining3 * 0.6
            lr2 = 0.005
            for _ in range(300):
                if time.time() > pg2_deadline:
                    break
                smooth_val, grad_pos, grad_ang = compute_smooth_outer_side_and_grad(
                    best_positions, best_angles, beta=80.0
                )
                new_pos = best_positions - lr2 * grad_pos
                new_ang = best_angles - lr2 * grad_ang * (180.0 / math.pi)
                
                angles_rad_new = np.radians(new_ang)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad_new[i])
                        h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                    if new_side < best_side - 1e-10:
                        best_positions = new_pos
                        best_angles = new_ang
                        best_side = new_side
                lr2 *= 0.97
    
    # Final fine-tune of outer hexagon angle
    remaining2 = DEADLINE - (time.time() - t0)
    if remaining2 > 3.0:
        def eval_final_phi(phi_deg):
            phi = np.radians(phi_deg)
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        try:
            res = minimize_scalar(eval_final_phi, bounds=(0, 60), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
                angles_rad = np.radians(best_angles)
                all_verts = get_all_vertices(best_positions, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                best_cx = center[0]
                best_cy = center[1]
        except Exception:
            pass
    
    # Validate and finalize
    if not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = make_grid_2x3(1.0)
    
    final_side, final_phi, final_cx, final_cy = compute_min_outer_side(best_positions, best_angles)
    if final_side < best_side:
        best_side = final_side
        best_phi = final_phi
        best_cx = final_cx
        best_cy = final_cy
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
