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
        # Max distance between centers for touching hexagons is 2*sqrt(3)
        if dist_sq > 12.0: # (2*sqrt(3))^2 = 12
            return False
        
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Vertices of h1
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        # Vertices of h2
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        # Normals of h1
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        # Normals of h2
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        # Separation axis theorem
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

    def compute_outer_side_and_center(positions, angles_deg, phi_init=None):
        """
        Computes the minimum side length of the outer hexagon containing all inner hexagons.
        Returns (side_length, center_x, center_y, optimal_rotation_deg)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Compute centroid
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Fully vectorized: compute side length for all 120 phi angles at once
        # phi_vals: 120 values from 0 to 59.5 (0.5° resolution)
        phi_vals = np.arange(0, 60, 0.5)
        phi_rad = np.radians(phi_vals)  # shape (120,)
        
        # For each phi (120 values), we have 6 normal directions
        # normal_angles[phi_idx, k] = phi_rad[phi_idx] + pi/6 + k*pi/3
        # Shape: (120, 6)
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)  # (120, 6)
        ny = np.sin(normal_angles)  # (120, 6)
        
        # offsets shape: (72, 2)
        # proj[phi_idx, vert_idx, k] = offsets[vert_idx, 0] * nx[phi_idx, k] + offsets[vert_idx, 1] * ny[phi_idx, k]
        # We want: max over vert_idx of |proj| for each (phi_idx, k)
        # Then: side[phi_idx] = max over k of (2/sqrt3 * max_proj[phi_idx, k])
        
        # offsets[:, 0:1] shape (72, 1, 1)
        # nx[None, :, :] shape (1, 120, 6)
        # Result: (72, 120, 6)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)  # (72, 120, 6)
        max_projs = np.max(abs_proj, axis=0)  # (120, 6)
        side_vals = (2.0 / sqrt3) * max_projs  # (120, 6)
        best_sides = np.max(side_vals, axis=1)  # (120,)
        
        best_idx = np.argmin(best_sides)
        best_side = best_sides[best_idx]
        best_phi = phi_vals[best_idx]
        
        # Refine the best angle using bounded minimize
        lo = max(0.0, best_phi - 0.5)
        hi = min(60.0, best_phi + 0.5)
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles_r = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx_r = np.cos(normal_angles_r)
            ny_r = np.sin(normal_angles_r)
            proj_r = offsets[:, 0:1] * nx_r[None, :] + offsets[:, 1:2] * ny_r[None, :]
            abs_proj_r = np.abs(proj_r)
            max_projs_r = np.max(abs_proj_r, axis=0)
            side_vals_r = (2.0 / sqrt3) * max_projs_r
            return np.max(side_vals_r)
            
        try:
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded', options={'xatol': 1e-10})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, center[0], center[1], best_phi

    def is_valid(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if check_collision(h1, h2):
                    return False
        return True

    def smooth_collision_penalty(pos, ang_rad, beta=30.0):
        """
        Smooth penalty for collisions using soft-min of SAT gaps.
        Returns 0 if no collisions, positive value if overlapping.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 12.0:
                    continue
                
                a1 = ang_rad[i]
                a2 = ang_rad[j]
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)
                
                v1x = pos[i, 0] + _ux * c1 - _uy * s1
                v1y = pos[i, 1] + _ux * s1 + _uy * c1
                v2x = pos[j, 0] + _ux * c2 - _uy * s2
                v2y = pos[j, 1] + _ux * s2 + _uy * c2
                
                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2
                
                min_gap = float('inf')
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gap = min(p2.max() - p1.min(), p1.max() - p2.min())
                    if gap < min_gap:
                        min_gap = gap
                    if min_gap <= 0:
                        break
                if min_gap > 0:
                    continue
                for k in range(6):
                    nx, ny = en2x[k], en2y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gap = min(p2.max() - p1.min(), p1.max() - p2.min())
                    if gap < min_gap:
                        min_gap = gap
                    if min_gap <= 0:
                        break
                
                if min_gap < 0:
                    penalty += -min_gap
        return penalty

    def smooth_outer_side(pos, ang_deg, phi_deg, beta=20.0):
        """
        Smooth approximation of the outer hexagon side length using log-sum-exp.
        """
        ang_rad = np.radians(ang_deg)
        all_verts = get_all_vertices(pos, ang_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # proj shape: (72, 6)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)  # (72, 6)
        
        # Log-sum-exp over vertices for each of 6 directions
        max_per_dir = np.max(abs_proj, axis=0)  # (6,)
        logsumexp_per_dir = max_per_dir + np.log(np.sum(np.exp(abs_proj - max_per_dir[:, None]), axis=0))  # (6,)
        
        # Log-sum-exp over 6 directions
        max_ls = np.max(logsumexp_per_dir)
        logsumexp_all = max_ls + np.log(np.sum(np.exp(logsumexp_per_dir - max_ls)))
        
        side = (2.0 / sqrt3) * logsumexp_all
        return side

    def smooth_objective(x, beta=30.0, phi_beta=20.0):
        """
        Combined smooth objective: outer side + collision penalty.
        x = [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, phi]
        Last element is outer hexagon rotation.
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = x[i*3]
            pos[i, 1] = x[i*3+1]
            ang[i] = x[i*3+2]
        phi = x[N*3]
        
        ang_rad = np.radians(ang)
        
        # Collision penalty
        coll_pen = smooth_collision_penalty(pos, ang_rad, beta)
        
        # Outer side (smooth)
        outer = smooth_outer_side(pos, ang, phi, phi_beta)
        
        # Total: outer side + large penalty for collisions
        total = outer + 50.0 * coll_pen
        return total

    # --- Initialization: Honeycomb ---
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_cx = 0.0
    best_cy = 0.0
    best_phi = 0.0

    # --- Phase 1: Simulated Annealing ---
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_cx, current_cy, current_phi = compute_outer_side_and_center(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_cx = current_cx
        best_cy = current_cy
        best_phi = current_phi

    T = 0.5
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.3
    step_decay = 0.99998
    
    sa_time_budget = DEADLINE * 0.35
    max_steps = 100000
    step_count = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
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
            if check_collision(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005
            continue
            
        new_side, new_cx, new_cy, new_phi = compute_outer_side_and_center(new_positions, new_angles)
        
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
        if step_size < 0.005: step_size = 0.005

    # --- Phase 2: Coordinate Descent Polish ---
    
    def polish_coordinate_descent(pos, ang, time_limit):
        current_side, _, _, _ = compute_outer_side_and_center(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()
        
        t_start = time.time()
        step = 0.1
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
                            if check_collision(h_moved, h_other):
                                valid = False
                                break
                            
                        if not valid:
                            continue
                            
                        new_side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                        
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
                    continue
                break
            
            if not improved:
                step *= 0.7
            else:
                step *= 0.95
                if step < min_step:
                    step = min_step
            
        return best_side_cd, best_pos_cd, best_ang_cd

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 10.0:
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, remaining_time * 0.30)
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
        
        # Phase 3: L-BFGS-B with smooth surrogate from multiple restarts
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 8.0:
            n_params = N * 3 + 1  # x, y, angle for each hexagon + outer phi
            
            def run_lbfgsb_restart(x0, time_budget):
                """Run L-BFGS-B from a single starting point."""
                t_start = time.time()
                
                def obj(x):
                    if time.time() - t_start > time_budget:
                        return 1e8
                    return smooth_objective(x, beta=30.0, phi_beta=20.0)
                
                try:
                    res = minimize(
                        obj, x0, method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
                    )
                    return res
                except Exception:
                    return None
            
            # Build initial x from current best
            def make_x(pos, ang, phi):
                x = np.zeros(N * 3 + 1)
                for i in range(N):
                    x[i*3] = pos[i, 0]
                    x[i*3+1] = pos[i, 1]
                    x[i*3+2] = ang[i]
                x[N*3] = phi
                return x
            
            x0_base = make_x(best_positions, best_angles, best_phi)
            
            # Try multiple restarts: base, perturbed, and from SA current
            restarts = [x0_base.copy()]
            
            # Perturbed versions
            for r in range(4):
                x_pert = x0_base.copy()
                for i in range(N):
                    x_pert[i*3] += random.gauss(0, 0.3)
                    x_pert[i*3+1] += random.gauss(0, 0.3)
                    x_pert[i*3+2] += random.gauss(0, 3.0)
                x_pert[N*3] += random.gauss(0, 5.0)
                restarts.append(x_pert)
            
            # From SA current state
            x_sa = make_x(current_positions, current_angles, current_phi)
            restarts.append(x_sa)
            
            per_restart_time = remaining_time * 0.50 / len(restarts)
            
            for xi, x_start in enumerate(restarts):
                if time.time() - t0 > DEADLINE - 3.0:
                    break
                
                res = run_lbfgsb_restart(x_start, per_restart_time)
                
                if res is None:
                    continue
                
                x_opt = res.x
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = x_opt[i*3]
                    new_pos[i, 1] = x_opt[i*3+1]
                    new_ang[i] = x_opt[i*3+2]
                new_phi = x_opt[N*3]
                
                # Verify validity with exact collision check
                angles_rad = np.radians(new_ang)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                        h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                        if check_collision(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_side, new_cx, new_cy, new_phi_exact = compute_outer_side_and_center(new_pos, new_ang)
                
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_pos.copy()
                    best_angles = new_ang.copy()
                    best_cx = new_cx
                    best_cy = new_cy
                    best_phi = new_phi_exact
            
            # Final coordinate descent polish after L-BFGS-B
            remaining_time = DEADLINE - (time.time() - t0)
            if remaining_time > 3.0:
                cd_side2, cd_pos2, cd_ang2 = polish_coordinate_descent(best_positions, best_angles, remaining_time * 0.8)
                if cd_side2 < best_side:
                    best_side = cd_side2
                    best_positions = cd_pos2
                    best_angles = cd_ang2
                    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)

    # Final verification
    if not is_valid(best_positions, best_angles):
        # Fallback to base if something went wrong
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
