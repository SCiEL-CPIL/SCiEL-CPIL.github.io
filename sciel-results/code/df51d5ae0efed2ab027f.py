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
        
        # Vectorized coarse scan: evaluate 120 angles (0.5° step) at once
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)
        side_vals = (2.0 / sqrt3) * max_projs
        best_idx = np.argmin(np.max(side_vals, axis=1))
        best_side = np.max(side_vals[best_idx])
        best_phi = float(best_idx * 0.5)
        
        # Fine tune with minimize_scalar
        try:
            lo = max(0.0, best_phi - 0.5)
            hi = min(60.0, best_phi + 0.5)
            
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

    def make_honeycomb():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    def make_ring_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            a = k * np.pi / 3
            positions[k, 0] = 2*sqrt3 * math.cos(a)
            positions[k, 1] = 2*sqrt3 * math.sin(a)
        positions[6] = [0, 0]
        for k in range(5):
            a = k * 2*np.pi/5 + np.pi/5
            positions[7+k, 0] = 3.5*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3.5*sqrt3 * math.sin(a)
        return positions, angles

    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
            positions[k+6, 0] = k * 2*sqrt3 + sqrt3/2
            positions[k+6, 1] = 3.0
        return positions, angles

    def make_triangular():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(3):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
        for k in range(4):
            positions[3+k, 0] = k * 2*sqrt3 + sqrt3/2
            positions[3+k, 1] = 3.0
        for k in range(3):
            positions[7+k, 0] = k * 2*sqrt3
            positions[7+k, 1] = 6.0
        return positions, angles

    def make_compact_hex():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        for k in range(6):
            a = k * np.pi / 3
            positions[1+k, 0] = 2*sqrt3 * math.cos(a)
            positions[1+k, 1] = 2*sqrt3 * math.sin(a)
        for k in range(5):
            a = k * 2*np.pi/5
            positions[7+k, 0] = 3*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3*sqrt3 * math.sin(a)
        return positions, angles

    def make_random_cluster():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        for k in range(1, N):
            a = random.uniform(0, 2*np.pi)
            r = random.uniform(1.5, 4.0)
            positions[k, 0] = r * math.cos(a)
            positions[k, 1] = r * math.sin(a)
            angles[k] = random.uniform(-30, 30)
        return positions, angles

    # Evaluate multiple initial configurations
    inits = [make_honeycomb(), make_ring_config(), make_two_rows(), make_triangular(), make_compact_hex()]
    # Add a few random inits
    for _ in range(5):
        inits.append(make_random_cluster())
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos_init, ang_init in inits:
        if not is_valid_configuration(pos_init, ang_init):
            continue
        s, phi, cx, cy = compute_min_outer_side(pos_init, ang_init)
        if s < best_side:
            best_side = s
            best_positions = pos_init.copy()
            best_angles = ang_init.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    # If no valid config found, use honeycomb as fallback
    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # === DIFFERENTIAL EVOLUTION PHASE ===
    # DE is a global optimizer that works well for this kind of packing problem
    de_time_budget = DEADLINE * 0.60
    
    def de_objective(params):
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
                    return 1e4 + 10 * sum(1 for i2 in range(N) for j2 in range(i2+1, N)
                                          if not hexagons_disjoint_fast(
                                              (pos[i2,0], pos[i2,1], angles_rad[i2]),
                                              (pos[j2,0], pos[j2,1], angles_rad[j2])))
        
        side, _, _, _ = compute_min_outer_side(pos, ang)
        return side

    # Initialize DE population
    pop_size = 30
    dim = 3 * N
    F = 0.8  # mutation factor
    CR = 0.9  # crossover rate
    
    # Start from best solution + random perturbations
    pop = np.zeros((pop_size, dim))
    for i in range(pop_size):
        for j in range(N):
            pop[i, 3*j] = best_positions[j, 0] + random.gauss(0, 0.5)
            pop[i, 3*j+1] = best_positions[j, 1] + random.gauss(0, 0.5)
            pop[i, 3*j+2] = best_angles[j] + random.gauss(0, 10.0)
    
    # Evaluate initial population
    pop_fit = np.array([de_objective(pop[i]) for i in range(pop_size)])
    best_idx = np.argmin(pop_fit)
    best_de_side = pop_fit[best_idx]
    best_de_pos = None
    best_de_ang = None
    
    if best_de_side < 1e4:
        best_de_pos = np.zeros((N, 2))
        best_de_ang = np.zeros(N)
        for j in range(N):
            best_de_pos[j, 0] = pop[best_idx, 3*j]
            best_de_pos[j, 1] = pop[best_idx, 3*j+1]
            best_de_ang[j] = pop[best_idx, 3*j+2]
        if best_de_side < best_side:
            best_side = best_de_side
            best_positions = best_de_pos.copy()
            best_angles = best_de_ang.copy()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    de_start = time.time()
    max_de_iter = 5000
    
    for it in range(max_de_iter):
        if time.time() - t0 > de_time_budget:
            break
        
        for i in range(pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = random.sample([k for k in range(pop_size) if k != i], 3)
            
            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = np.zeros(dim)
            j_rand = random.randint(0, dim - 1)
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]
            
            # Selection
            trial_fit = de_objective(trial)
            if trial_fit <= pop_fit[i]:
                pop[i] = trial
                pop_fit[i] = trial_fit
                if trial_fit < best_de_side:
                    best_de_side = trial_fit
                    best_de_pos = np.zeros((N, 2))
                    best_de_ang = np.zeros(N)
                    for j in range(N):
                        best_de_pos[j, 0] = trial[3*j]
                        best_de_pos[j, 1] = trial[3*j+1]
                        best_de_ang[j] = trial[3*j+2]
                    if best_de_side < best_side and best_de_side < 1e4:
                        best_side = best_de_side
                        best_positions = best_de_pos.copy()
                        best_angles = best_de_ang.copy()
                        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # === POLISH PHASE ===
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 3.0:
        # Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.03
            min_step = 0.0005
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim_i in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_cd.copy()
                            new_ang = best_ang_cd.copy()
                            
                            if dim_i == 0:
                                new_pos[idx, 0] += direction * step
                            elif dim_i == 1:
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
                        continue
                    break
                else:
                    step *= 0.8
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.4, 15.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # Nelder-Mead polish on smooth surrogate
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
            def smooth_objective(params, beta=80.0):
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
                
                all_verts = get_all_vertices(pos, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                offsets = verts_flat - center
                
                best_smooth = float('inf')
                for phi_deg in np.arange(0, 60, 0.5):
                    phi = np.radians(phi_deg)
                    normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                    nx = np.cos(normal_angles)
                    ny = np.sin(normal_angles)
                    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                    abs_proj = np.abs(proj)
                    max_projs = np.max(abs_proj, axis=0)
                    side_vals = (2.0 / sqrt3) * max_projs
                    
                    m = np.max(side_vals)
                    smooth_max = m + math.log(np.sum(np.exp(side_vals - m)))
                    if smooth_max < best_smooth:
                        best_smooth = smooth_max
                
                return best_smooth

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                nm_time = min(remaining_time * 0.3, 10.0)
                res = minimize(smooth_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                
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
        
        # Final high-precision coordinate descent
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 2.0:
            final_cd_time = min(remaining_time * 0.5, 8.0)
            final_side, final_pos, final_ang = polish_coordinate_descent(best_positions, best_angles, final_cd_time)
            if final_side < best_side:
                best_side = final_side
                best_positions = final_pos
                best_angles = final_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos_f, ang_f in inits:
            if is_valid_configuration(pos_f, ang_f):
                best_positions = pos_f.copy()
                best_angles = ang_f.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
