import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 142.0  # Safety margin before 150s wall-clock kill

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
    
    # Precompute for speed
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

    def check_collision(h1, h2):
        """
        Returns True if h1 and h2 overlap.
        h1, h2 are (x, y, angle_rad)
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 12.0:
            return False
        
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
                return False
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
        return True

    def compute_outer_side_and_center(positions, angles_deg, refine=True):
        """
        Computes the minimum side length of the outer hexagon containing all inner hexagons.
        Returns (side_length, center_x, center_y, optimal_rotation_deg)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_vals = np.arange(0, 60, 0.25)
        phi_rad = np.radians(phi_vals)
        
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        best_sides = np.max(side_vals, axis=1)
        
        best_idx = np.argmin(best_sides)
        best_side = best_sides[best_idx]
        best_phi = phi_vals[best_idx]
        
        if refine:
            lo = max(0.0, best_phi - 0.25)
            hi = min(60.0, best_phi + 0.25)
            
            def eval_phi(phi_deg):
                phi = np.radians(phi_deg)
                na = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx_r = np.cos(na)
                ny_r = np.sin(na)
                proj_r = offsets[:, 0:1] * nx_r[None, :] + offsets[:, 1:2] * ny_r[None, :]
                abs_proj_r = np.abs(proj_r)
                max_projs_r = np.max(abs_proj_r, axis=0)
                side_vals_r = (2.0 / sqrt3) * max_projs_r
                return np.max(side_vals_r)
                
            try:
                res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded', options={'xatol': 1e-12})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
            except Exception:
                pass
        
        return best_side, center[0], center[1], best_phi

    def fast_side_length(positions, angles_deg):
        """Fast side-length evaluation without refine (for SA loop)."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_vals = np.arange(0, 60, 0.5)
        phi_rad = np.radians(phi_vals)
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        best_sides = np.max(side_vals, axis=1)
        best_idx = np.argmin(best_sides)
        return best_sides[best_idx], center[0], center[1], phi_vals[best_idx]

    def smooth_objective(x, temperature=0.03):
        """
        Smooth surrogate objective using log-sum-exp for the max projection.
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = x[i*3]
            pos[i, 1] = x[i*3+1]
            ang[i] = x[i*3+2]
        
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_vals = np.arange(0, 60, 1.0)
        phi_rad = np.radians(phi_vals)
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        
        soft_max_dirs = np.zeros(60)
        for i in range(60):
            vals = side_vals[i, :]
            m = np.max(vals)
            soft_max_dirs[i] = m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))
        
        neg_vals = -soft_max_dirs
        m = np.max(neg_vals)
        soft_min_phi = -(m + temperature * np.log(np.sum(np.exp((neg_vals - m) / temperature))))
        
        smooth_side = -soft_min_phi
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 12.0:
                    continue
                
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
                
                gaps = []
                for i_ax in range(6):
                    nx, ny = en1x[i_ax], en1y[i_ax]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gap = p2.min() - p1.max()
                    gaps.append(gap)
                for i_ax in range(6):
                    nx, ny = en2x[i_ax], en2y[i_ax]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gap = p2.min() - p1.max()
                    gaps.append(gap)
                
                gaps_arr = np.array(gaps)
                min_gap = np.min(gaps_arr)
                if min_gap < 0:
                    penalty += 50.0 * (-min_gap) ** 2
        
        return smooth_side + penalty

    def is_valid(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if check_collision(h1, h2):
                    return False
        return True

    # --- Initialization: Multiple starting configurations ---
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

    def make_compact_hex():
        # Center + 6 around + 5 outer
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            pos[i+1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i+7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_radial_cluster():
        """Radial clustering: 1 center, 6 inner ring, 5 outer ring with tighter spacing"""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        # Inner ring: 6 hexagons touching center
        for i in range(6):
            theta = i * np.pi / 3
            r = 2 * sqrt3 * 0.95  # Slightly compressed
            pos[i+1] = [r * math.cos(theta), r * math.sin(theta)]
        # Outer ring: 5 hexagons
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            r = 4 * sqrt3 * 0.88  # Compressed outer ring
            pos[i+7] = [r * math.cos(theta), r * math.sin(theta)]
        return pos, ang

    def make_tight_ring():
        """Tight two-ring configuration"""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # 6 in inner ring
        for i in range(6):
            theta = i * np.pi / 3
            r = 2 * sqrt3 * 0.92
            pos[i] = [r * math.cos(theta), r * math.sin(theta)]
        # 6 in outer ring
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            r = 4 * sqrt3 * 0.85
            pos[i+6] = [r * math.cos(theta), r * math.sin(theta)]
        return pos, ang

    def make_triangular_cluster():
        """Triangular arrangement"""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # 3x3 triangular grid minus corners, then adjust
        h = 1.5  # vertical spacing
        w = sqrt3  # horizontal spacing
        coords = [
            (0, 0), (w, 0), (2*w, 0),
            (w/2, h), (3*w/2, h),
            (0, 2*h), (w, 2*h), (2*w, 2*h),
            (w/2, 3*h), (3*w/2, 3*h),
            (w, 4*h),
            (w/2, -h),
        ]
        for idx, (x, y) in enumerate(coords):
            pos[idx] = [x - 1.5*w, y - 2*h]
        return pos, ang

    init_configs = [make_honeycomb(), make_ring_config(), make_compact_hex(), 
                    make_radial_cluster(), make_tight_ring(), make_triangular_cluster()]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_cx = 0.0
    best_cy = 0.0
    best_phi = 0.0

    for pos0, ang0 in init_configs:
        if is_valid(pos0, ang0):
            s, cx, cy, phi = compute_outer_side_and_center(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_cx = cx
                best_cy = cy
                best_phi = phi

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)

    # --- Phase 1: Simulated Annealing ---
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_cx, current_cy, current_phi = fast_side_length(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_cx = current_cx
        best_cy = current_cy
        best_phi = current_phi

    T = 0.3
    T_min = 0.00005
    cooling = 0.99997
    step_size = 0.25
    step_decay = 0.99999
    
    sa_time_budget = DEADLINE * 0.40
    max_steps = 500000
    step_count = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
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
            if check_collision(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003
            continue
            
        new_side, new_cx, new_cy, new_phi = fast_side_length(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_cx = new_cx
            current_cy = new_cy
            current_phi = new_phi
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_cx = new_cx
                best_cy = new_cy
                best_phi = new_phi
        else:
            delta = new_side - current_side
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_cx = new_cx
                current_cy = new_cy
                current_phi = new_phi
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_cx = new_cx
                    best_cy = new_cy
                    best_phi = new_phi
        
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003

    # --- Phase 1.5: Smooth Surrogate Optimization with L-BFGS-B ---
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 25.0:
        x0 = np.zeros(N*3)
        for i in range(N):
            x0[i*3] = best_positions[i, 0]
            x0[i*3+1] = best_positions[i, 1]
            x0[i*3+2] = best_angles[i]
        
        best_x = x0.copy()
        best_val = smooth_objective(x0, 0.03)
        
        t_start = time.time()
        
        # Try multiple temperature values for the smooth surrogate
        for temp in [0.15, 0.08, 0.04, 0.02, 0.01]:
            if time.time() - t_start > remaining_time * 0.75:
                break
            
            def obj(x):
                return smooth_objective(x, temp)
            
            try:
                res = minimize(
                    obj,
                    best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if res.fun < best_val:
                    # Check if the result is actually valid (no collisions)
                    res_pos = np.zeros((N, 2))
                    res_ang = np.zeros(N)
                    for i in range(N):
                        res_pos[i, 0] = res.x[i*3]
                        res_pos[i, 1] = res.x[i*3+1]
                        res_ang[i] = res.x[i*3+2]
                    
                    angles_rad = np.radians(res_ang)
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            h1 = (res_pos[i, 0], res_pos[i, 1], angles_rad[i])
                            h2 = (res_pos[j, 0], res_pos[j, 1], angles_rad[j])
                            if check_collision(h1, h2):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_x = res.x.copy()
                        best_val = res.fun
                    else:
                        # If invalid, try a smaller penalty by reducing temperature
                        pass
            
            except Exception:
                pass
        
        # Extract best result from smooth optimization
        best_pos = np.zeros((N, 2))
        best_ang = np.zeros(N)
        for i in range(N):
            best_pos[i, 0] = best_x[i*3]
            best_pos[i, 1] = best_x[i*3+1]
            best_ang[i] = best_x[i*3+2]
        
        if is_valid(best_pos, best_ang):
            smooth_side, smooth_cx, smooth_cy, smooth_phi = compute_outer_side_and_center(best_pos, best_ang)
            if smooth_side < best_side:
                best_side = smooth_side
                best_positions = best_pos
                best_angles = best_ang
                best_cx = smooth_cx
                best_cy = smooth_cy
                best_phi = smooth_phi

    # --- Phase 2: Nelder-Mead Local Refinement ---
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 15.0:
        def nm_objective(x):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = x[i*3]
                pos[i, 1] = x[i*3+1]
                ang[i] = x[i*3+2]
            
            angles_rad = np.radians(ang)
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                    h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                    if check_collision(h1, h2):
                        return 100.0
            
            side, _, _, _ = fast_side_length(pos, ang)
            return side
        
        x0 = np.zeros(N*3)
        for i in range(N):
            x0[i*3] = best_positions[i, 0]
            x0[i*3+1] = best_positions[i, 1]
            x0[i*3+2] = best_angles[i]
        
        try:
            nm_result = minimize(
                nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if nm_result.fun < 100.0:
                nm_pos = np.zeros((N, 2))
                nm_ang = np.zeros(N)
                for i in range(N):
                    nm_pos[i, 0] = nm_result.x[i*3]
                    nm_pos[i, 1] = nm_result.x[i*3+1]
                    nm_ang[i] = nm_result.x[i*3+2]
                
                if nm_result.fun < best_side:
                    best_positions = nm_pos
                    best_angles = nm_ang
                    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
        except Exception:
            pass
        
        # Phase 3: Coordinate Descent final polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 12.0:
            def polish_coordinate_descent(pos, ang, time_limit):
                current_side, _, _, _ = compute_outer_side_and_center(pos, ang)
                best_side_cd = current_side
                best_pos_cd = pos.copy()
                best_ang_cd = ang.copy()
                
                t_start = time.time()
                step = 0.03
                min_step = 0.000005
                
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
                                    if check_collision(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                new_side, _, _, _ = fast_side_length(new_pos, new_ang)
                                
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
                            continue
                        break
                    
                    if not improved:
                        step *= 0.6
                    else:
                        step *= 0.92
                        if step < min_step:
                            step = min_step
                
                return best_side_cd, best_pos_cd, best_ang_cd
            
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, remaining_time * 0.85)
            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
                best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)

    # Final verification
    if not is_valid(best_positions, best_angles):
        best_positions, best_angles = make_honeycomb()
        best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    # Final high-precision polish with multi-stage coordinate descent
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 8.0:
        # Multi-stage coordinate descent with adaptive steps
        stages = [(0.005, 40), (0.002, 50), (0.0008, 60), (0.0003, 80), (0.0001, 100), (0.00003, 120)]
        for step, max_iters in stages:
            if time.time() - t0 > DEADLINE - 3.0:
                break
            for _ in range(max_iters):
                if time.time() - t0 > DEADLINE - 3.0:
                    break
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
                                if check_collision(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                            
                            if new_side < best_side - 1e-14:
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
                    break
    
    # Recompute final center and phi
    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
