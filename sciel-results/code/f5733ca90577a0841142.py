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

def _soft_max(x, beta=50.0):
    """Smooth max via log-sum-exp."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _soft_min(x, beta=50.0):
    """Smooth min via negative log-sum-exp."""
    return -_soft_max(-x, beta)

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
    
    for s in [0.95, 0.98, 1.0, 1.02, 1.05]:
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
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Evaluate all initial configs
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
    
    # SA phase
    sa_time_budget = DEADLINE * 0.45
    sa_deadline = t0 + sa_time_budget
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    T = 0.3
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.15
    step_decay = 0.99999
    
    while time.time() < sa_deadline:
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 3.0)
        
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
            if delta < 0: delta = 0
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
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003
    
    # Gradient-based optimization with smooth surrogate
    remaining = DEADLINE - (time.time() - t0)
    
    if remaining > 15.0:
        grad_deadline = t0 + remaining * 0.6
        
        def compute_smooth_obj_and_grad(x, beta=30.0):
            """
            x: [2N] positions + [N] angles (degrees)
            Returns (objective, gradient)
            """
            pos = x[:2*N].reshape(N, 2)
            ang_deg = x[2*N:]
            ang_rad = np.radians(ang_deg)
            
            # Compute vertices
            all_verts = get_all_vertices(pos, ang_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            
            # Compute smooth outer side length
            # Try multiple phi values and take soft max
            phi_candidates = np.radians(np.arange(0, 60, 3))
            n_phi = len(phi_candidates)
            
            # For each phi, compute the 6 side values
            # side_vals shape: (n_phi, 6)
            normal_angles = phi_candidates[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)  # (n_phi, 6)
            ny = np.sin(normal_angles)
            
            # proj: (72, n_phi, 6)
            proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)  # (n_phi, 6)
            side_vals = (2.0 / sqrt3) * max_projs
            
            # Soft max over all phi and sides
            obj = _soft_max(side_vals.flatten(), beta=beta)
            
            # Gradient computation
            grad = np.zeros_like(x)
            
            # For the smooth max, we need the gradient w.r.t. each vertex position
            # This is complex, so use numerical gradient for positions and angles
            # But that's slow. Instead, use a simpler approach:
            # Just compute the objective and let L-BFGS-B handle it numerically
            # Actually, let's just return the objective and let minimize handle gradients numerically
            
            return obj, None
        
        # Use a simpler smooth objective that's easier to differentiate
        def compute_smooth_obj(x, beta=30.0):
            pos = x[:2*N].reshape(N, 2)
            ang_deg = x[2*N:]
            ang_rad = np.radians(ang_deg)
            
            all_verts = get_all_vertices(pos, ang_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            
            # Use a single best phi estimate
            phi_vals = np.radians(np.arange(0, 60, 5))
            normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            max_sides = np.max(side_vals, axis=1)
            
            # Soft min over phi candidates
            return _soft_min(max_sides, beta=beta)
        
        # Start from current best
        x0 = np.concatenate([best_positions.flatten(), best_angles])
        
        try:
            res = minimize(
                compute_smooth_obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
            )
            
            if res.success:
                new_pos = res.x[:2*N].reshape(N, 2)
                new_ang = res.x[2*N:]
                
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos.copy()
                        best_angles = new_ang.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass
        
        # Try from perturbed starts
        if time.time() < grad_deadline:
            for restart in range(5):
                if time.time() > grad_deadline:
                    break
                x0_pert = x0.copy()
                x0_pert[:2*N] += random.gauss(0, 0.08, (2*N,))
                x0_pert[2*N:] += random.gauss(0, 1.5, (N,))
                try:
                    res = minimize(
                        compute_smooth_obj,
                        x0_pert,
                        method='L-BFGS-B',
                        options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
                    )
                    if res.success:
                        new_pos = res.x[:2*N].reshape(N, 2)
                        new_ang = res.x[2*N:]
                        if is_valid_configuration(new_pos, new_ang):
                            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                            if new_side < best_side:
                                best_side = new_side
                                best_positions = new_pos.copy()
                                best_angles = new_ang.copy()
                                best_phi = new_phi
                                best_cx = new_cx
                                best_cy = new_cy
                except Exception:
                    pass
    
    # Coordinate descent polish
    remaining = DEADLINE - (time.time() - t0)
    
    if remaining > 5.0:
        cd_deadline = t0 + remaining * 0.5
        step = 0.03
        min_step = 0.0001
        
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
                step *= 0.6
            else:
                step = max(step * 0.8, min_step)
        
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
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
