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

        # Vectorized coarse scan with finer resolution
        phi_vals = np.radians(np.arange(0, 60, 0.5))
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
        best_phi = float(best_idx * 0.5)

        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-9})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def compute_smooth_outer_side(positions, angles_deg, beta=60.0):
        """Smooth log-sum-exp surrogate for the outer side objective."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        def eval_smooth(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            # Log-sum-exp for smooth max over vertices
            scaled = beta * abs_proj
            max_scaled = np.max(scaled, axis=0)
            log_sum_exp = (1.0/beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled[:, None]), axis=0)))
            side_vals = (2.0 / sqrt3) * log_sum_exp
            return np.max(side_vals)

        # Coarse scan
        phi_vals = np.arange(0, 60, 1)
        best_side = float('inf')
        best_phi = 0.0
        for p in phi_vals:
            s = eval_smooth(p)
            if s < best_side:
                best_side = s
                best_phi = p

        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)
            res = minimize_scalar(eval_smooth, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-7})
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

    def make_honeycomb(scale=1.0, offset_angle=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N) + offset_angle
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles

    def make_ring_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N
            r = 2.5
            positions[i, 0] = r * math.cos(theta)
            positions[i, 1] = r * math.sin(theta)
            angles[i] = theta * 180 / math.pi
        return positions, angles

    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(6):
            positions[i, 0] = i * sqrt3
            positions[i, 1] = 0
            positions[i+6, 0] = (i + 0.5) * sqrt3
            positions[i+6, 1] = 1.5
        return positions, angles

    # --- NEW STRUCTURAL COMPONENTS: GEOMETRIC HEURISTIC SEARCH ---

    def relax_overlaps(positions, angles_rad, iterations=20):
        """
        Iteratively push overlapping hexagons apart.
        Returns modified positions and angles.
        """
        pos = positions.copy()
        ang = angles_rad.copy()
        
        for _ in range(iterations):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    if not hexagons_disjoint_fast((pos[i,0], pos[i,1], ang[i]), (pos[j,0], pos[j,1], ang[j])):
                        # Calculate center-to-center vector
                        dx = pos[j,0] - pos[i,0]
                        dy = pos[j,1] - pos[i,1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-9:
                            # Random nudge if centers coincide
                            dx = random.gauss(0, 0.1)
                            dy = random.gauss(0, 0.1)
                            dist = math.sqrt(dx*dx + dy*dy)
                        
                        # Minimum distance for disjointness is 2.0 (side length 1)
                        # Actually, for hexagons, min center dist is 2.0 if aligned.
                        # We push them to 2.1 to ensure separation.
                        target_dist = 2.1
                        if dist < target_dist:
                            # Normalize vector
                            nx = dx / dist
                            ny = dy / dist
                            # Push apart by half the required distance
                            push = (target_dist - dist) / 2.0
                            pos[i,0] -= nx * push
                            pos[i,1] -= ny * push
                            pos[j,0] += nx * push
                            pos[j,1] += ny * push
                            moved = True
            if not moved:
                break
        return pos, ang

    def geometric_insertion_search(max_time=10.0):
        """
        Greedy insertion of hexagons to minimize bounding box growth.
        """
        best_side = float('inf')
        best_pos = None
        best_ang = None
        
        # Candidate positions for the first hexagon (centered)
        # We will try inserting N-1 hexagons around a fixed center one.
        
        # Pre-compute candidate angles for new hexagons
        candidate_angles = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]) * np.pi / 180
        
        # Start with hexagon 0 at origin
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        
        # Try different starting configurations for the first few
        # Here we just start with one at (0,0) and insert others.
        
        t_start = time.time()
        
        # We will run a few restarts with different insertion orders or random perturbations
        for restart in range(5):
            if time.time() - t_start > max_time:
                break
            
            # Reset
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            
            # Place first hexagon
            pos[0, 0] = 0
            pos[0, 1] = 0
            ang[0] = 0
            
            # Insert remaining N-1
            for k in range(1, N):
                best_candidate_side = float('inf')
                best_candidate_pos = None
                best_candidate_ang = None
                
                # Candidate positions: Ring around existing hexagons
                # We sample points in a grid or ring around the current cluster
                # Simple strategy: Try positions relative to each existing hexagon
                candidates = []
                
                for i in range(k):
                    # Try 6 neighbors of hex i (honeycomb style)
                    for m in range(6):
                        theta = m * np.pi / 3
                        r = 2.0 # Distance for touching
                        # Add some jitter for variety
                        jitter_r = random.gauss(0, 0.1)
                        jitter_t = random.gauss(0, 0.05)
                        
                        cx = pos[i,0] + (r + jitter_r) * math.cos(theta + jitter_t)
                        cy = pos[i,1] + (r + jitter_r) * math.sin(theta + jitter_t)
                        
                        # Try a few angles for the new hexagon
                        for a_idx in range(len(candidate_angles)):
                            a = candidate_angles[a_idx] + random.gauss(0, 0.05)
                            candidates.append((cx, cy, a))
                
                # Also try some random positions in a larger area to escape local minima
                for _ in range(20):
                    r = random.uniform(1.0, 5.0)
                    theta = random.uniform(0, 2*np.pi)
                    cx = r * math.cos(theta)
                    cy = r * math.sin(theta)
                    a = random.uniform(0, 2*np.pi)
                    candidates.append((cx, cy, a))
                
                # Evaluate candidates
                for cx, cy, a in candidates:
                    # Tentatively place
                    pos[k,0] = cx
                    pos[k,1] = cy
                    ang[k] = a
                    
                    # Check validity (disjointness with all previous)
                    valid = True
                    h_new = (cx, cy, a)
                    for i in range(k):
                        h_old = (pos[i,0], pos[i,1], ang[i])
                        if not hexagons_disjoint_fast(h_new, h_old):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Compute side length
                    side, _, _, _ = compute_min_outer_side(pos[:k+1], np.degrees(ang[:k+1]))
                    
                    if side < best_candidate_side:
                        best_candidate_side = side
                        best_candidate_pos = (cx, cy)
                        best_candidate_ang = a
                
                if best_candidate_pos is None:
                    # If no valid candidate found, break or randomize
                    # This shouldn't happen often if we have enough candidates
                    break
                
                pos[k,0] = best_candidate_pos[0]
                pos[k,1] = best_candidate_pos[1]
                ang[k] = best_candidate_ang
            
            # Relax overlaps
            pos_relaxed, ang_relaxed = relax_overlaps(pos, ang, iterations=10)
            
            # Check validity
            if is_valid_configuration(pos_relaxed, np.degrees(ang_relaxed)):
                side, _, _, _ = compute_min_outer_side(pos_relaxed, np.degrees(ang_relaxed))
                if side < best_side:
                    best_side = side
                    best_pos = pos_relaxed.copy()
                    best_ang = ang_relaxed.copy()
        
        return best_side, best_pos, best_ang

    # --- MAIN SEARCH LOGIC ---

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Phase 1: Geometric Heuristic Search
    # Allocate 60% of time to this
    geo_time_budget = DEADLINE * 0.6
    geo_start = time.time()
    
    geo_side, geo_pos, geo_ang = geometric_insertion_search(max_time=geo_time_budget)
    
    if geo_pos is not None:
        if geo_side < best_side:
            best_side = geo_side
            best_positions = geo_pos
            best_angles = np.degrees(geo_ang)
            # Compute center and phi
            _, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Phase 2: Polish with L-BFGS-B and Coordinate Descent
    remaining_time = DEADLINE - (time.time() - t0)
    
    if best_positions is None:
        # Fallback to initial configs if geometric search failed
        init_pos, init_ang = make_honeycomb(1.0, 0.0)
        best_positions = init_pos
        best_angles = init_ang
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    if remaining_time > 5.0:
        # 1. L-BFGS-B with smooth surrogate
        if remaining_time > 10.0:
            def smooth_objective(params):
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

                side, _, _, _ = compute_smooth_outer_side(pos, ang, beta=80.0)
                return side

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]

            try:
                res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                               options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})

                if res.fun < 1e5:  # valid result
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

        # 2. Tight coordinate descent polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 6.0:
            def polish_coordinate_descent(pos, ang, time_limit):
                current_side, _, _, _ = compute_min_outer_side(pos, ang)
                best_side_cd = current_side
                best_pos_cd = pos.copy()
                best_ang_cd = ang.copy()

                t_start = time.time()
                step = 0.01
                min_step = 0.0001
                no_improve_count = 0
                max_no_improve = 5

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

                                if new_side < best_side_cd - 1e-12:
                                    best_pos_cd = new_pos
                                    best_ang_cd = new_ang
                                    best_side_cd = new_side
                                    improved = True
                                    no_improve_count = 0
                                    break

                            if time.time() - t_start > time_limit:
                                return best_side_cd, best_pos_cd, best_ang_cd
                        else:
                            continue
                        break
                    else:
                        no_improve_count += 1
                        if no_improve_count >= max_no_improve:
                            step *= 0.5
                            no_improve_count = 0
                        continue
                    break

                return best_side_cd, best_pos_cd, best_ang_cd

            cd_time = min(remaining_time * 0.5, 20.0)
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)

            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

        # 3. Second L-BFGS-B pass
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 8.0:
            def smooth_objective2(params):
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

                side, _, _, _ = compute_smooth_outer_side(pos, ang, beta=100.0)
                return side

            init_params2 = np.zeros(3*N)
            for i in range(N):
                init_params2[3*i] = best_positions[i, 0]
                init_params2[3*i+1] = best_positions[i, 1]
                init_params2[3*i+2] = best_angles[i]

            try:
                res2 = minimize(smooth_objective2, init_params2, method='L-BFGS-B',
                                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})

                if res2.fun < 1e5:
                    new_pos2 = np.zeros((N, 2))
                    new_ang2 = np.zeros(N)
                    for i in range(N):
                        new_pos2[i, 0] = res2.x[3*i]
                        new_pos2[i, 1] = res2.x[3*i+1]
                        new_ang2[i] = res2.x[3*i+2]

                    if is_valid_configuration(new_pos2, new_ang2):
                        new_side2, new_phi2, new_cx2, new_cy2 = compute_min_outer_side(new_pos2, new_ang2)
                        if new_side2 < best_side:
                            best_side = new_side2
                            best_positions = new_pos2
                            best_angles = new_ang2
                            best_phi = new_phi2
                            best_cx = new_cx2
                            best_cy = new_cy2
            except Exception:
                pass

        # 4. Final high-precision minimize_scalar
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            def eval_phi_final(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)

            try:
                lo = max(0.0, best_phi - 5.0)
                hi = min(60.0, best_phi + 5.0)
                res = minimize_scalar(eval_phi_final, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-12, 'maxiter': 500})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
                    best_cx = center[0]
                    best_cy = center[1]
            except Exception:
                pass

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to a known valid config if something went wrong
        init_pos, init_ang = make_honeycomb(1.0, 0.0)
        best_positions = init_pos
        best_angles = init_ang
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
