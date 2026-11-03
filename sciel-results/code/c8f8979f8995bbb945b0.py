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
        
        # Coarse scan first to find a good region
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune with minimize_scalar
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

    # Generate diverse initial configurations
    def generate_honeycomb(seed_offset=0):
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        patterns = [
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(2,0),(-2,0),(1,-1),(-1,1),(0,2)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,-1),(-2,-1),(1,-1),(-1,-1),(0,2)],
        ]
        idx = (seed_offset) % len(patterns)
        lattice_coords = patterns[idx]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i, (li, lj) in enumerate(lattice_coords):
            positions[i] = li * a1 + lj * a2
            angles[i] = 0.0
        return positions, angles

    def generate_random_cluster():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        angles[0] = random.uniform(0, 60)
        
        for i in range(1, N):
            for attempt in range(50):
                ref_idx = random.randint(0, i-1)
                dx = random.uniform(-3.5, 3.5)
                dy = random.uniform(-3.5, 3.5)
                positions[i] = positions[ref_idx] + [dx, dy]
                angles[i] = random.uniform(0, 60)
                
                valid = True
                angles_rad = np.radians(angles[:i+1])
                h_new = (positions[i, 0], positions[i, 1], angles_rad[i])
                for j in range(i):
                    h_other = (positions[j, 0], positions[j, 1], angles_rad[j])
                    if not hexagons_disjoint_fast(h_new, h_other):
                        valid = False
                        break
                if valid:
                    break
            
            if not valid:
                positions[i] = [random.uniform(-5, 5), random.uniform(-5, 5)]
                angles[i] = random.uniform(0, 60)
        
        return positions, angles

    def generate_ring_pattern():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        angles[0] = 0
        for i in range(1, 7):
            ang = i * 60 * np.pi / 180
            dist = 3.0
            positions[i] = [dist * np.cos(ang), dist * np.sin(ang)]
            angles[i] = 0
        for i in range(7, 12):
            ang = (i - 7) * 72 * np.pi / 180 + 30 * np.pi / 180
            dist = 5.5
            positions[i] = [dist * np.cos(ang), dist * np.sin(ang)]
            angles[i] = 0
        return positions, angles

    def smooth_objective(params):
        """
        Smooth objective using log-sum-exp for max projection and penalty for overlap.
        """
        if time.time() - t0 > DEADLINE:
            return 1e6

        pos = np.zeros((N, 2))
        ang_deg = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang_deg[i] = params[3*i+2]
        
        angles_rad = np.radians(ang_deg)
        
        # 1. Separation Penalty
        # For each pair, compute minimum distance (approx via vertex distance or support function)
        # A simpler penalty: sum of squared overlaps.
        # Let's use a soft penalty based on distance between centers minus a threshold.
        # Actually, a proper smooth approximation of the separation constraint is better.
        # Let's use the sum of exp(k * (dist - min_dist)) where dist is center distance? No, that's not accurate for hexagons.
        # Let's stick to a simpler penalty: if centers are too close, penalize.
        # Better: Use the existing disjoint check logic but smoothed? 
        # Since we are using L-BFGS-B, we need a differentiable objective.
        # Let's use a penalty term that grows when hexagons overlap.
        # Approximation: Sum over pairs of max(0, (R1+R2 - dist_centers))^2 is not differentiable at 0, but smoothable.
        # Even better: Use the support function distance.
        
        # For now, a simple penalty: 
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Hexagons of side 1 have "radius" (center to vertex) 1.
                # Minimum distance to touch is sqrt(3) ~ 1.732 (flat-to-flat) or 2.0 (vertex-to-vertex)?
                # Actually, if aligned, flat-to-flat distance is sqrt(3). 
                # Let's assume a minimum center distance of 1.732 * 2 = 3.464? No.
                # If two unit hexagons touch flat-to-flat, center distance is sqrt(3).
                # If vertex-to-vertex, it's 2.
                # Let's use a threshold of 1.732 (sqrt(3)) as the minimum possible distance.
                min_dist = math.sqrt(3)
                if dist < min_dist:
                    # Smooth penalty
                    overlap = min_dist - dist
                    penalty += overlap * overlap
                    # Add a gradient-like term to push them apart? 
                    # L-BFGS-B handles gradients via finite diff if not provided.
                    # Providing gradient is faster.
        
        # 2. Containment Objective (Smooth)
        # Compute min outer side length.
        # We can't easily make the "min over phi" smooth without differentiating the inner minimization.
        # Strategy: Fix phi to the current best estimate or optimize phi as part of the vector?
        # If we include phi in the optimization vector, we have 3N + 1 variables.
        # Let's include phi in the vector.
        # params[3*N] = phi_deg
        
        phi_deg = params[3*N]
        phi = np.radians(phi_deg)
        
        # Get all vertices
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Center of mass of all vertices
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Projection onto 6 normals
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        
        # We want to minimize the max of (2/sqrt(3) * max(abs_proj[:, k]))
        # Let M_k = max(abs_proj[:, k])
        # Side_k = (2/sqrt(3)) * M_k
        # We want min_phi max_k Side_k(phi)
        
        # Log-sum-exp approximation of max_k Side_k
        # Let S_k = (2/sqrt(3)) * M_k
        # log(sum(exp(beta * S_k))) / beta approximates max(S_k)
        
        # First, approximate M_k = max_i abs_proj[i,k] using log-sum-exp
        beta_proj = 50.0
        # abs_proj shape (72, 6)
        # M_k = log(sum_i exp(beta * abs_proj[i,k])) / beta
        # To avoid overflow, subtract max
        max_abs_proj = np.max(abs_proj, axis=0) # (6,)
        exp_terms = np.exp(beta_proj * (abs_proj - max_abs_proj[None, :]))
        sum_exp = np.sum(exp_terms, axis=0)
        M_k_smooth = (np.log(sum_exp) + beta_proj * max_abs_proj) / beta_proj
        
        S_k = (2.0 / sqrt3) * M_k_smooth
        
        # Now approximate max_k S_k
        beta_side = 50.0
        max_S = np.max(S_k)
        exp_S = np.exp(beta_side * (S_k - max_S))
        sum_exp_S = np.sum(exp_S)
        max_side_smooth = (np.log(sum_exp_S) + beta_side * max_S) / beta_side
        
        total_obj = max_side_smooth + 10.0 * penalty
        
        return total_obj

    def run_gradient_optimization(positions, angles, time_budget):
        init_params = np.zeros(3*N + 1)
        for i in range(N):
            init_params[3*i] = positions[i, 0]
            init_params[3*i+1] = positions[i, 1]
            init_params[3*i+2] = angles[i]
        
        # Initial phi estimate
        side, phi, cx, cy = compute_min_outer_side(positions, angles)
        init_params[3*N] = phi
        
        t_start = time.time()
        
        try:
            # L-BFGS-B
            res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 5000})
            
            if res.fun < 100.0: # Valid range check
                new_params = res.x
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = new_params[3*i]
                    new_pos[i, 1] = new_params[3*i+1]
                    new_ang[i] = new_params[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    return new_side, new_pos, new_ang, new_phi, new_cx, new_cy
        except Exception:
            pass
        
        return None

    def polish_coordinate_descent(pos, ang, phi, time_limit):
        """Refine using coordinate descent on positions and angles, keeping phi fixed or re-optimizing."""
        current_side, current_phi, cx, cy = compute_min_outer_side(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()
        best_phi_cd = current_phi
        
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
                        
                        new_side, new_phi, _, _ = compute_min_outer_side(new_pos, new_ang)
                        
                        if new_side < best_side_cd - 1e-10:
                            best_pos_cd = new_pos
                            best_ang_cd = new_ang
                            best_side_cd = new_side
                            best_phi_cd = new_phi
                            improved = True
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                step *= 0.7
                continue
            break
        
        return best_side_cd, best_pos_cd, best_ang_cd, best_phi_cd

    # Portfolio approach
    candidates = []
    for offset in range(3):
        pos, ang = generate_honeycomb(offset)
        candidates.append((pos, ang, "honeycomb"))
    pos, ang = generate_ring_pattern()
    candidates.append((pos, ang, "ring"))
    for seed in range(5):
        random.seed(seed * 12345)
        pos, ang = generate_random_cluster()
        candidates.append((pos, ang, f"random_{seed}"))
        
    for i, (pos, ang, name) in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 10:
            break
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        candidates[i] = (pos, ang, name, side, phi)
        
    candidates.sort(key=lambda x: x[3])
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    remaining_time = DEADLINE - (time.time() - t0) - 15.0
    if remaining_time < 5:
        remaining_time = 5
        
    time_per_candidate = remaining_time / max(1, len(candidates))
    
    for i, (pos, ang, name, init_side, init_phi) in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 15:
            break
        
        sa_time = min(time_per_candidate, remaining_time * 0.4)
        
        if init_side < best_side + 1.0 or best_positions is None:
            res = run_gradient_optimization(pos, ang, sa_time)
            if res:
                g_side, g_pos, g_ang, g_phi, g_cx, g_cy = res
                if g_side < best_side:
                    best_side = g_side
                    best_positions = g_pos
                    best_angles = g_ang
                    best_phi = g_phi
                    best_cx = g_cx
                    best_cy = g_cy
                    
        remaining_time = DEADLINE - (time.time() - t0) - 15.0
        if remaining_time < 5:
            break
            
    # Final Polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0 and best_positions is not None:
        cd_time = min(remaining_time * 0.5, 10.0)
        cd_side, cd_pos, cd_ang, cd_phi = polish_coordinate_descent(best_positions, best_angles, best_phi, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_phi = cd_phi
            # Recalculate center
            _, _, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
            
    if best_positions is None or not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = generate_honeycomb(0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
