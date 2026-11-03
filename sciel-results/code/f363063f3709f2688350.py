"""
Combined approach: SA + scipy polish with vectorized overlap check and smooth outer side evaluation.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill

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

    # Strategy: SA for 65% of the time, then Polish for the rest
    sa_time_budget = DEADLINE * 0.65
    
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.5
    T_min = 0.0001
    cooling = 0.9999
    step_size = 0.3
    step_decay = 0.99998
    
    step_count = 0
    max_steps = 200000
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
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005
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
        if step_size < 0.005: step_size = 0.005

    # Polish phase: use scipy.optimize.minimize with SLSQP
    # Objective: outer side length + penalty for overlaps
    def polish_objective(x):
        positions = x[:2*N].reshape(N, 2)
        angles_deg = x[2*N:]
        
        # Check validity and compute penalty
        angles_rad = np.radians(angles_deg)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                # Compute minimum distance violation
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 4.5:
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
                    # SAT: find minimum separation
                    min_sep = float('inf')
                    for i_n in range(6):
                        nx, ny = en1x[i_n], en1y[i_n]
                        p1 = v1x * nx + v1y * ny
                        p2 = v2x * nx + v2y * ny
                        sep = min(p2.min() - p1.max(), p1.min() - p2.max())
                        if sep < 0:
                            min_sep = min(min_sep, -sep)
                    for i_n in range(6):
                        nx, ny = en2x[i_n], en2y[i_n]
                        p1 = v1x * nx + v1y * ny
                        p2 = v2x * nx + v2y * ny
                        sep = min(p2.min() - p1.max(), p1.min() - p2.max())
                        if sep < 0:
                            min_sep = min(min_sep, -sep)
                    if min_sep < float('inf'):
                        penalty += min_sep * min_sep
        
        # Compute outer side
        side, phi, cx, cy = compute_min_outer_side(positions, angles_deg)
        return side + 10.0 * penalty
    
    # Start polishing from best found
    x0 = np.concatenate([best_positions.flatten(), best_angles])
    
    # Use SLSQP with tight tolerance
    try:
        result = minimize(polish_objective, x0, method='SLSQP', 
                         options={'maxiter': 50, 'ftol': 1e-10})
        
        if result.success:
            new_positions = result.x[:2*N].reshape(N, 2)
            new_angles = result.x[2*N:]
            
            # Verify validity
            if is_valid_configuration(new_positions, new_angles):
                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
    except Exception:
        pass
    
    # Final high-precision polish: refine phi for the best configuration
    if best_side < float('inf'):
        try:
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            
            def eval_phi_precise(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)
            
            # Fine scan
            best_phi_final = best_phi
            best_side_final = best_side
            for phi_deg in np.arange(max(0, best_phi - 0.5), min(60, best_phi + 0.5), 0.01):
                s = eval_phi_precise(phi_deg)
                if s < best_side_final:
                    best_side_final = s
                    best_phi_final = phi_deg
            
            # Even finer with minimize_scalar
            lo = max(0.0, best_phi_final - 0.01)
            hi = min(60.0, best_phi_final + 0.01)
            res = minimize_scalar(eval_phi_precise, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_side_final:
                best_side_final = res.fun
                best_phi_final = res.x
            
            best_side = best_side_final
            best_phi = best_phi_final
            best_cx = center[0]
            best_cy = center[1]
        except Exception:
            pass

    # Final Result
    inner = [[float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])] for i in range(N)]
    return inner, (float(best_cx), float(best_cy)), float(best_side), float(best_phi)

# EVOLVE-BLOCK-END
