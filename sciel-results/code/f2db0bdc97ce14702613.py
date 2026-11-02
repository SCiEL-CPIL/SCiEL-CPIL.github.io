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
# Edge normals: for each edge from vert i to vert (i+1)%6
_edge_normals_raw = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _edge_normals_raw[_i] = [_nx / _norm, _ny / _norm]
_EDGE_NORMALS = _edge_normals_raw  # (6, 2) unit normals for edges of unit hex at angle 0

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    
    # Precompute for fast SAT: vertices and normals as flat arrays
    _ux = unit_hex_verts[:, 0]  # (6,)
    _uy = unit_hex_verts[:, 1]  # (6,)
    _enx = edge_normals[:, 0]   # (6,)
    _eny = edge_normals[:, 1]   # (6,)
    
    # Helper: Get vertices of all hexagons given positions and angles
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        return all_verts

    def hexagons_disjoint_fast(h1, h2):
        """
        Fast SAT check for two unit hexagons.
        h1, h2: (x, y, angle_rad)
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        if dist_sq > 4.5:
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Compute vertices
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        # Rotate edge normals
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        # Check all 12 normals using vectorized projections
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
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Returns (side_length, optimal_outer_angle_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        
        # Flatten to (72, 2)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Center
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        sqrt3 = math.sqrt(3)
        
        # Vectorized evaluation for a given phi
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            # 6 normal directions for the outer hexagon
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)  # (6,)
            ny = np.sin(normal_angles)  # (6,)
            
            # Project all vertices onto each normal: (72, 6)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)  # (72, 6)
            
            # For each direction, get max absolute projection
            max_projs = np.max(abs_proj, axis=0)  # (6,)
            side_vals = (2.0 / sqrt3) * max_projs  # (6,)
            return np.max(side_vals)
        
        # Use minimize_scalar for precision
        res = minimize_scalar(eval_phi, bounds=(0, 60), method='bounded', options={'xatol': 1e-5})
        if res.success:
            best_phi = res.x
            best_side = res.fun
        else:
            # Fallback to coarse scan
            best_side = float('inf')
            best_phi = 0.0
            for phi_deg in range(0, 60, 1):
                s = eval_phi(phi_deg)
                if s < best_side:
                    best_side = s
                    best_phi = phi_deg

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        """Check if all hexagons are pairwise disjoint."""
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Initial Configuration: Honeycomb packing
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    
    lattice_coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
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
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Run SA for a few different initial perturbations
    num_restarts = 4
    sa_time_budget = DEADLINE * 0.75  # Reserve 25% for polish
    
    for r in range(num_restarts):
        if time.time() - t0 > sa_time_budget:
            break
            
        # Perturb initial configuration
        current_positions = base_positions.copy()
        current_angles = base_angles.copy()
        if r > 0:
            current_positions += np.random.normal(0, 0.15, size=(N, 2))
            current_angles += np.random.normal(0, 8.0, size=N)
            
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        
        if current_side < best_side:
            best_side = current_side
            best_positions = current_positions.copy()
            best_angles = current_angles.copy()
            best_phi = current_phi
            best_cx = current_cx
            best_cy = current_cy

        T = 0.3
        T_min = 0.001
        cooling = 0.99985
        step_size = 0.25
        step_decay = 0.99995
        
        step_count = 0
        max_steps = 50000
        accept_count = 0
        total_count = 0
        no_improve = 0
        
        while time.time() - t0 < sa_time_budget and step_count < max_steps:
            step_count += 1
            total_count += 1
            
            idx = random.randint(0, N - 1)
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            da = random.gauss(0, 6.0) # degrees
            
            new_positions[idx, 0] += dx
            new_positions[idx, 1] += dy
            new_angles[idx] += da
            
            # Check validity
            angles_rad_new = np.radians(new_angles)
            valid = True
            h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
            for j in range(N):
                if j == idx: continue
                h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h_moved, h_other):
                    valid = False
                    break
                    
            if not valid:
                no_improve += 1
                T *= cooling
                if T < T_min: T = T_min
                step_size *= step_decay
                if step_size < 0.01: step_size = 0.01
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
                no_improve = 0
                
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
                else:
                    no_improve += 1
                        
            # Adaptive step size: increase if acceptance rate too low, decrease if too high
            if total_count % 500 == 0 and total_count > 0:
                acc_rate = accept_count / total_count
                if acc_rate < 0.1:
                    step_size *= 0.7
                    if step_size < 0.01: step_size = 0.01
                elif acc_rate > 0.5:
                    step_size *= 1.3
                    if step_size > 1.0: step_size = 1.0
                accept_count = 0
                total_count = 0
            
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.01: step_size = 0.01

    # Polish Step: Coordinate descent for faster convergence
    def polish_coordinate_descent(pos, ang, time_limit):
        """
        Coordinate descent polish: move one hexagon at a time to reduce outer side.
        """
        current_side, _, _, _ = compute_min_outer_side(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()
        
        t_start = time.time()
        step = 0.05
        min_step = 0.001
        
        while time.time() - t_start < time_limit and step > min_step:
            improved = False
            for idx in range(N):
                # Try moving in x, y, and angle
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
                        
                        # Check validity
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
                            break  # Break direction loop for this dim
                    
                    if time.time() - t_start > time_limit:
                        return best_side_cd, best_pos_cd, best_ang_cd
                else:
                    continue
                break
            else:
                # No improvement in full pass, reduce step
                step *= 0.8
                continue
            break
        
        return best_side_cd, best_pos_cd, best_ang_cd

    # Polish Step
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
        
    polish_time_budget = DEADLINE - (time.time() - t0)
    if polish_time_budget > 5.0:
        try:
            # First: coordinate descent polish
            cd_time = min(polish_time_budget * 0.4, 20.0)
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
            
            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
            
            # Then: Nelder-Mead for final fine-tuning
            remaining_time = DEADLINE - (time.time() - t0)
            if remaining_time > 5.0:
                init_params2 = np.zeros(3*N)
                for i in range(N):
                    init_params2[3*i] = best_positions[i, 0]
                    init_params2[3*i+1] = best_positions[i, 1]
                    init_params2[3*i+2] = best_angles[i]
                
                res = minimize(polish_objective, init_params2, method='Nelder-Mead', 
                               options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8})
                
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

    # Final verification and return
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to base configuration if somehow invalid
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
