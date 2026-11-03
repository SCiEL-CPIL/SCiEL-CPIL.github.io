import math
import time
import random
import numpy as np
from scipy.optimize import minimize

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

    # Honeycomb initial configuration
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
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Generate diverse initial configurations
    def generate_configurations():
        configs = []
        # 1. Base honeycomb
        configs.append((base_positions.copy(), base_angles.copy()))
        
        # 2. Slightly perturbed honeycomb
        for _ in range(5):
            pos = base_positions.copy() + np.random.normal(0, 0.2, (N, 2))
            ang = base_angles.copy() + np.random.normal(0, 10.0, N)
            configs.append((pos, ang))
            
        # 3. Random cluster
        for _ in range(5):
            pos = np.random.normal(0, 3.0, (N, 2))
            ang = np.random.uniform(-180, 180, N)
            configs.append((pos, ang))
            
        return configs

    # Differentiable objective for L-BFGS-B
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles_deg = params[2*N:]
        angles_rad = np.radians(angles_deg)
        
        # Compute vertices
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # 1. Containment penalty (log-sum-exp smooth max of distances)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        dists = np.linalg.norm(offsets, axis=1)
        # Approximate outer hexagon radius as max distance
        # Smooth max: logsumexp(dists / T) * T
        T = 0.1
        max_dist = T * np.log(np.sum(np.exp(dists / T)))
        
        # 2. Separation penalty
        # For each pair, compute minimum distance
        sep_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Approximate distance between hexagons using centers and radius
                # More accurate: check if vertices are inside each other?
                # For gradient descent, a simpler penalty based on center distance is often sufficient
                # if the hexagons are roughly circular.
                # However, to be robust, we can use a penalty if center distance < 2*radius
                # Radius of unit hexagon is 1.
                # So if dist < 2, they might overlap.
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Penalty if dist < 2.0 (they overlap if dist < 2 for circles, but hexagons can be closer)
                # Actually, two unit hexagons can be closer than 2 if rotated.
                # Minimum distance between two unit hexagons is 0 (touching).
                # A safer penalty: if dist < 1.5, penalize.
                # Let's use a smooth ramp.
                if dist < 1.8:
                    sep_penalty += (1.8 - dist) ** 2
        
        # 3. Objective: Minimize max_dist + lambda * sep_penalty
        # We want to minimize the outer hexagon side length.
        # side = (2/sqrt3) * max_proj
        # max_proj is related to max_dist.
        # Let's just minimize max_dist + penalty.
        
        return max_dist + 100.0 * sep_penalty

    # Strategy: Multi-start L-BFGS-B
    configs = generate_configurations()
    
    # Sort configs by initial objective to try best first
    def quick_eval(pos, ang):
        side, _, _, _ = compute_min_outer_side(pos, ang)
        return side
        
    scored_configs = []
    for pos, ang in configs:
        if not is_valid_configuration(pos, ang):
            continue
        s, phi, cx, cy = compute_min_outer_side(pos, ang)
        scored_configs.append((s, pos, ang, phi, cx, cy))
        
    # Update best with best initial config
    if scored_configs:
        scored_configs.sort(key=lambda x: x[0])
        best_s, best_p, best_a, best_phi, best_cx, best_cy = scored_configs[0]
        best_side = best_s
        best_positions = best_p
        best_angles = best_a
        best_phi = best_phi
        best_cx = best_cx
        best_cy = best_cy

    # Run L-BFGS-B on top few initial configs
    num_starts = min(5, len(scored_configs))
    
    for i in range(num_starts):
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        init_s, init_p, init_a, _, _, _ = scored_configs[i]
        
        init_params = np.zeros(3*N)
        for k in range(N):
            init_params[2*k] = init_p[k, 0]
            init_params[2*k+1] = init_p[k, 1]
            init_params[2*N+k] = init_a[k]
            
        try:
            res = minimize(objective, init_params, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < init_s + 100: # Heuristic check
                new_params = res.x
                new_p = np.zeros((N, 2))
                new_a = np.zeros(N)
                for k in range(N):
                    new_p[k, 0] = new_params[2*k]
                    new_p[k, 1] = new_params[2*k+1]
                    new_a[k] = new_params[2*N+k]
                
                # Verify validity
                if is_valid_configuration(new_p, new_a):
                    new_s, new_phi, new_cx, new_cy = compute_min_outer_side(new_p, new_a)
                    if new_s < best_side:
                        best_side = new_s
                        best_positions = new_p
                        best_angles = new_a
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
                        
        except Exception:
            pass

    # Final Polish with Coordinate Descent (exact, no penalty)
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        def polish_coordinate_descent(pos, ang, time_limit):
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
                    step *= 0.8
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.8, 20.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
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

