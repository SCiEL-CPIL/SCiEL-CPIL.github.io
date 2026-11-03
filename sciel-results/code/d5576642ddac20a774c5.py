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

    # ---------- Initial topologies ----------
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])

    # Topology 1: Honeycomb cluster
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    honeycomb_pos = np.zeros((N, 2))
    honeycomb_ang = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        honeycomb_pos[idx] = i * a1 + j * a2
        honeycomb_ang[idx] = 0.0

    # Topology 2: Ring of 12 around center
    ring_pos = np.zeros((N, 2))
    ring_ang = np.zeros(N)
    ring_radius = 2.2
    for idx in range(N):
        ang = 2 * math.pi * idx / N
        ring_pos[idx] = [ring_radius * math.cos(ang), ring_radius * math.sin(ang)]
        ring_ang[idx] = ang * 180 / math.pi

    # Topology 3: Two concentric rings (6 + 6)
    ring2_pos = np.zeros((N, 2))
    ring2_ang = np.zeros(N)
    for idx in range(6):
        ang = 2 * math.pi * idx / 6
        ring2_pos[idx] = [1.6 * math.cos(ang), 1.6 * math.sin(ang)]
        ring2_ang[idx] = 0.0
    for idx in range(6, 12):
        ang = 2 * math.pi * (idx - 6) / 6 + math.pi / 6
        ring2_pos[idx] = [3.0 * math.cos(ang), 3.0 * math.sin(ang)]
        ring2_ang[idx] = 0.0

    # Topology 4: Spiral
    spiral_pos = np.zeros((N, 2))
    spiral_ang = np.zeros(N)
    for idx in range(N):
        r = 0.3 + 0.35 * idx
        ang = 0.8 * idx
        spiral_pos[idx] = [r * math.cos(ang), r * math.sin(ang)]
        spiral_ang[idx] = ang * 180 / math.pi

    topologies = [
        (honeycomb_pos, honeycomb_ang),
        (ring_pos, ring_ang),
        (ring2_pos, ring2_ang),
        (spiral_pos, spiral_ang),
    ]

    # ---------- Smooth objective for gradient optimization ----------
    # We minimize: outer_side + lambda * overlap_penalty
    # Overlap penalty: sum over pairs of max(0, min_dist - dist)^2 where min_dist is
    # a conservative separation threshold. For unit hexagons, the minimum center
    # distance for non-overlap depends on orientation; we use a smooth proxy based
    # on the maximum of support functions.

    def smooth_objective(params, lam):
        """
        params: 3N array [x0,y0,a0, x1,y1,a1, ...] angles in radians
        Returns (value, grad) for L-BFGS-B
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]

        # Compute all vertices
        all_verts = get_all_vertices(pos, ang)  # shape (N, 6, 2)
        verts_flat = all_verts.reshape(-1, 2)

        # Center of mass
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        # Outer hexagon side: min over rotation phi of max projection
        # We use a fixed set of candidate rotations and take max, but for
        # smoothness we use the average of top-k or just a fixed optimal-ish phi.
        # For gradient purposes, we use a smooth approximation: the side length
        # is determined by the maximum projection onto 6 directions. We use a
        # log-sum-exp style smooth max over the 6 directions and 72 vertices.
        
        # For each of 6 outer hexagon edge normals (at optimal phi), the side
        # is (2/sqrt3) * max_i |dot(offset_i, n)|
        # We approximate the min over phi by using a fixed phi that we optimize
        # separately, or by using a smooth max over a range of phis.
        
        # Simpler: use the convex hull width approach. The minimum enclosing
        # regular hexagon side is related to the maximum width in 6 directions.
        # We use a fixed set of 12 directions (every 30 deg) and take the max
        # projection, then convert to side length. This is an upper bound but
        # smooth.
        
        dir_angles = np.pi/12 + np.arange(12) * np.pi/6  # 12 directions, 30 deg apart
        nx = np.cos(dir_angles)
        ny = np.sin(dir_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)  # shape (12,)
        # The side length for a regular hexagon enclosing these projections:
        # For a regular hexagon with side s, the width in direction theta is
        # s * (sqrt(3)/2) * (1 + cos(2*(theta - phi))) for appropriate phi.
        # We take the max over the 12 directions and convert.
        # A regular hexagon of side s has width sqrt(3)*s in the "flat" direction
        # and 2*s in the "pointy" direction. The width varies between sqrt(3)*s
        # and 2*s. We use: s = max_proj / (sqrt(3)/2) for the flat case, but
        # this is an upper bound. For a tighter bound, we use the minimum over
        # rotations. For smoothness, we use the average of the 12 max_projs
        # scaled appropriately.
        
        # Actually, the minimum enclosing regular hexagon side is:
        # s = (2/sqrt(3)) * min_phi max_i |dot(offset_i, n(phi+i*60deg))|
        # We approximate min_phi by using a fixed phi that we'll optimize.
        # For the gradient step, we fix phi to 0 and use a smooth max.
        
        # Let's use a different approach: the side length is determined by the
        # maximum of the 6 support functions. We use a smooth max (log-sum-exp)
        # over the 6 directions for a fixed phi, and we'll optimize phi separately.
        
        phi = 0.0  # fixed for now, will optimize separately
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx6 = np.cos(normal_angles)
        ny6 = np.sin(normal_angles)
        proj6 = offsets[:, 0:1] * nx6[None, :] + offsets[:, 1:2] * ny6[None, :]
        abs_proj6 = np.abs(proj6)
        max_projs6 = np.max(abs_proj6, axis=0)  # shape (6,)
        
        # Smooth max of the 6 values
        beta = 50.0
        smooth_max = np.log(np.sum(np.exp(beta * max_projs6))) / beta
        side_len = (2.0 / sqrt3) * smooth_max

        # Overlap penalty: for each pair, compute the minimum distance between
        # the two hexagons using support functions. We use a smooth proxy:
        # For two convex polygons, the distance is min over all edge pairs of
        # the separation. We use a simpler proxy: the distance between centers
        # minus the maximum of the two "radii" in the direction between centers.
        # This is not exact but smooth and provides a useful penalty.
        
        overlap_penalty = 0.0
        grad_penalty = np.zeros_like(params)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    continue
                ux = dx / dist
                uy = dy / dist
                
                # Maximum extent of hex i in direction (ux, uy)
                c1 = math.cos(ang[i])
                s1 = math.sin(ang[i])
                rx1 = _ux * c1 - _uy * s1
                ry1 = _ux * s1 + _uy * c1
                proj1 = rx1 * ux + ry1 * uy
                max1 = np.max(proj1)
                
                # Maximum extent of hex j in direction (-ux, -uy)
                c2 = math.cos(ang[j])
                s2 = math.sin(ang[j])
                rx2 = _ux * c2 - _uy * s2
                ry2 = _ux * s2 + _uy * c2
                proj2 = rx2 * (-ux) + ry2 * (-uy)
                max2 = np.max(proj2)
                
                # Separation: dist - max1 - max2
                # If negative, they overlap (approximately)
                sep = dist - max1 - max2
                if sep < 0:
                    # Smooth penalty: (max(0, -sep))^2
                    overlap_penalty += (-sep) ** 2
                    # Gradient with respect to positions and angles
                    dsep_ddx = ux
                    dsep_ddy = uy
                    # d(max1)/d(pos[i]) = -ux, -uy (since max1 is extent from center i)
                    # d(max1)/d(ang[i]) = d(rx1*ux + ry1*uy)/d(ang)
                    # This is complex; we use a simpler gradient approximation
                    # by finite differences if needed, or just position gradients.
                    # For now, we use position-only gradients for the penalty.
                    grad_penalty[3*i] -= 2 * (-sep) * dsep_ddx * (-1)  # d/dx_i of -sep = -dsep/dx_i
                    grad_penalty[3*i+1] -= 2 * (-sep) * dsep_ddy * (-1)
                    grad_penalty[3*j] += 2 * (-sep) * dsep_ddx
                    grad_penalty[3*j+1] += 2 * (-sep) * dsep_ddy
        
        value = side_len + lam * overlap_penalty
        return value, grad_penalty * lam

    def optimize_topology(pos0, ang0_deg, time_budget, lam_init=10.0):
        """
        Optimize starting from pos0, ang0_deg using L-BFGS-B with penalty.
        Returns (best_side, best_pos, best_ang_deg, best_phi, best_cx, best_cy)
        """
        t_start = time.time()
        
        # Convert angles to radians for the optimizer
        ang0_rad = np.radians(ang0_deg)
        params0 = np.zeros(3*N)
        for i in range(N):
            params0[3*i] = pos0[i, 0]
            params0[3*i+1] = pos0[i, 1]
            params0[3*i+2] = ang0_rad[i]
        
        best_side = float('inf')
        best_params = params0.copy()
        
        # Adaptive penalty: start high, decrease as overlaps clear
        lam = lam_init
        for it in range(20):
            if time.time() - t_start > time_budget * 0.8:
                break
            
            def obj_and_grad(p):
                val, grad = smooth_objective(p, lam)
                return val, grad
            
            try:
                res = minimize(obj_and_grad, best_params, jac=True, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
                if res.fun < best_side + 1e-6:
                    # Check validity and compute true side
                    new_pos = np.zeros((N, 2))
                    new_ang_rad = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang_rad[i] = res.x[3*i+2]
                    
                    new_ang_deg = np.degrees(new_ang_rad)
                    
                    if is_valid_configuration(new_pos, new_ang_deg):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang_deg)
                        if new_side < best_side:
                            best_side = new_side
                            best_params = res.x.copy()
                        else:
                            # Keep the better of the two
                            if res.fun < best_side:
                                best_params = res.x.copy()
                    else:
                        # Invalid, but keep params for next iteration with higher penalty
                        best_params = res.x.copy()
                else:
                    best_params = res.x.copy()
            except Exception:
                pass
            
            # Decrease penalty if overlaps are small
            val, _ = smooth_objective(best_params, lam)
            side_only, _, _, _ = compute_min_outer_side(
                np.array([[best_params[3*i], best_params[3*i+1]] for i in range(N)]),
                np.degrees(np.array([best_params[3*i+2] for i in range(N)])))
            overlap_part = (val - side_only) / lam if lam > 0 else 0
            if overlap_part < 0.01:
                lam *= 0.5
            elif overlap_part > 1.0:
                lam *= 2.0
            else:
                lam *= 0.8
        
        # Final evaluation
        final_pos = np.zeros((N, 2))
        final_ang_rad = np.zeros(N)
        for i in range(N):
            final_pos[i, 0] = best_params[3*i]
            final_pos[i, 1] = best_params[3*i+1]
            final_ang_rad[i] = best_params[3*i+2]
        final_ang_deg = np.degrees(final_ang_rad)
        
        if is_valid_configuration(final_pos, final_ang_deg):
            final_side, final_phi, final_cx, final_cy = compute_min_outer_side(final_pos, final_ang_deg)
            if final_side < best_side:
                best_side = final_side
                best_params = best_params.copy()
        
        return best_side, final_pos, final_ang_deg, best_params

    # ---------- Run optimization on all topologies ----------
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    time_per_topo = (DEADLINE - 10.0) / len(topologies)  # leave 10s for polish

    for pos0, ang0 in topologies:
        if time.time() - t0 > DEADLINE - 15.0:
            break
        
        side, pos, ang, _ = optimize_topology(pos0, ang0, time_per_topo)
        
        if side < best_side:
            best_side = side
            best_positions = pos.copy()
            best_angles = ang.copy()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # ---------- Polish Phase ----------
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 3.0 and best_positions is not None:
        # Coordinate descent micro-polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.02
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
                                break
                        
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.7
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.6, 20.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

        # Final Nelder-Mead on the 3N params
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
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
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
                
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

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to honeycomb
        best_positions = honeycomb_pos.copy()
        best_angles = honeycomb_ang.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
