import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 143.0  # Safety margin before 150s wall-clock kill

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

        # Vectorized coarse scan: evaluate all 60 angles at once
        phi_vals = np.radians(np.arange(0, 60, 1))  # (60,)
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3  # (60, 6)
        nx = np.cos(normal_angles)  # (60, 6)
        ny = np.sin(normal_angles)  # (60, 6)
        # proj[i, j, k] = offset_j dot normal_i_k
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)  # (72, 60, 6)
        max_projs = np.max(abs_proj, axis=0)  # (60, 6)
        side_vals = (2.0 / sqrt3) * max_projs  # (60, 6)
        side_per_phi = np.max(side_vals, axis=1)  # (60,)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        # Fine tune with minimize_scalar
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()

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

    # Generate multiple initial configurations
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])

    def make_honeycomb():
        # Proper honeycomb: center + 6 ring + 5 partial second ring
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Center hexagon
        pos[0] = [0, 0]
        # First ring: 6 hexagons at distance 2*sqrt(3)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        # Second ring: 5 more hexagons (partial ring to make 12 total)
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
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

    def make_two_rows():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i] = [i * 2 * sqrt3, 0]
            pos[i + 6] = [i * 2 * sqrt3 + sqrt3, 3.0]
        return pos, ang

    def make_compact_hex():
        """Tighter hexagonal arrangement with center"""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    # Add a random perturbation config
    def make_random_config():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Start from honeycomb and add small random perturbations
        base_pos, base_ang = make_honeycomb()
        for i in range(N):
            pos[i] = base_pos[i] + np.array([random.gauss(0, 0.3), random.gauss(0, 0.3)])
            ang[i] = random.gauss(0, 10.0)
        return pos, ang

    init_configs = [make_honeycomb(), make_ring_config(), make_two_rows(), make_compact_hex(), make_random_config()]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Evaluate all initial configs
    for pos0, ang0 in init_configs:
        if is_valid_configuration(pos0, ang0):
            s, phi, cx, cy = compute_min_outer_side(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Strategy: Run SA for 55% of the time, then polish for the rest
    sa_time_budget = DEADLINE * 0.55

    # Fast approximate outer side (skip phi optimization, use fixed phi)
    def fast_outer_side(positions, angles_deg, phi_rad):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        na = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
        nxx = np.cos(na)
        nyy = np.sin(na)
        pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
        return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()

    # SA run with adaptive parameters
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    current_phi_rad = np.radians(current_phi)

    T = 0.3
    T_min = 0.0001
    cooling = 0.9999
    step_size = 0.25
    step_decay = 0.99998

    step_count = 0
    max_steps = 500000
    accept_count = 0
    total_count = 0
    full_eval_interval = 50  # Do full compute_min_outer_side every N steps

    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1

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
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break

        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003
            continue

        # Use fast evaluation most of the time, full evaluation periodically
        if step_count % full_eval_interval == 0:
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
            current_phi_rad = np.radians(new_phi)
        else:
            new_side = fast_outer_side(new_positions, new_angles, current_phi_rad)
            new_phi = current_phi
            new_cx = current_cx
            new_cy = current_cy

        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            if step_count % full_eval_interval == 0:
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
            accept_count += 1
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                if step_count % full_eval_interval == 0:
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
                if step_count % full_eval_interval == 0:
                    current_phi = new_phi
                    current_cx = new_cx
                    current_cy = new_cy
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    if step_count % full_eval_interval == 0:
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy

        # Adaptive step size
        if total_count % 200 == 0:
            acc_rate = accept_count / max(1, total_count)
            if acc_rate < 0.08:
                step_size *= 0.8
                if step_size < 0.003: step_size = 0.003
            elif acc_rate > 0.4:
                step_size *= 1.15
                if step_size > 0.8: step_size = 0.8
            accept_count = 0
            total_count = 0

        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # Contact-network refinement using support function inequalities
        def contact_network_refine(pos, ang, time_limit):
            """Iteratively refine using linearized non-overlap constraints at contact normals.
            
            For each pair of hexagons, the non-overlap condition can be expressed as:
            for all edge normals n of hexagon i: max_j (v_ij · n) <= min_k (v_jk · n)
            where v_ij are vertices of hexagon i and v_jk are vertices of hexagon j.
            
            We linearize around the current configuration and solve a QP-like problem
            using projected gradient descent on the side length objective.
            """
            t_start = time.time()
            current_pos = pos.copy()
            current_ang = ang.copy()
            current_side, current_phi, _, _ = compute_min_outer_side(current_pos, current_ang)
            current_phi_rad = np.radians(current_phi)
            
            # Precompute current vertices and normals for all hexagons
            def get_hex_data(pos_arr, ang_arr):
                """Return vertices and edge normals for each hexagon."""
                angles_rad = np.radians(ang_arr)
                c = np.cos(angles_rad)
                s = np.sin(angles_rad)
                # vertices: (N, 6, 2)
                vx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
                vy = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
                verts = np.stack((vx + pos_arr[:, 0:1], vy + pos_arr[:, 1:2]), axis=-1)
                # edge normals: (N, 6, 2)
                enx = _enx[None, :] * c[:, None] - _eny[None, :] * s[:, None]
                eny = _enx[None, :] * s[:, None] + _eny[None, :] * c[:, None]
                normals = np.stack((enx, eny), axis=-1)
                return verts, normals
            
            # Compute pairwise penetration depth and contact normal
            def pairwise_contact(pos_arr, ang_arr, i, j):
                """Return (penetration, normal_angle) for pair (i,j).
                penetration > 0 means overlap; normal is direction to push apart."""
                h1 = (pos_arr[i, 0], pos_arr[i, 1], np.radians(ang_arr[i]))
                h2 = (pos_arr[j, 0], pos_arr[j, 1], np.radians(ang_arr[j]))
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.5:
                    return 0.0, 0.0
                
                c1, s1 = math.cos(h1[2]), math.sin(h1[2])
                c2, s2 = math.cos(h2[2]), math.sin(h2[2])
                v1x = h1[0] + _ux * c1 - _uy * s1
                v1y = h1[1] + _ux * s1 + _uy * c1
                v2x = h2[0] + _ux * c2 - _uy * s2
                v2y = h2[1] + _ux * s2 + _uy * c2
                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2
                
                min_pen = float('inf')
                best_nx, best_ny = 0.0, 0.0
                
                # Check all 12 separating axes
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    # Penetration: how much they overlap along this axis
                    # If p1.max() < p2.min(): separated (no overlap on this axis)
                    # Otherwise: overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    # But we want the minimum over all axes that show overlap
                    pen = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    if pen < min_pen:
                        min_pen = pen
                        best_nx, best_ny = nx, ny
                
                for k in range(6):
                    nx, ny = en2x[k], en2y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    pen = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    if pen < min_pen:
                        min_pen = pen
                        best_nx, best_ny = nx, ny
                
                if min_pen <= 0:
                    return 0.0, 0.0
                
                return min_pen, math.atan2(best_ny, best_nx)
            
            # Iterative refinement
            step = 0.02
            min_step = 0.00001
            max_iters = 5000
            
            for it in range(max_iters):
                if time.time() - t_start > time_limit:
                    break
                
                # Compute current side
                current_side, current_phi, _, _ = compute_min_outer_side(current_pos, current_ang)
                current_phi_rad = np.radians(current_phi)
                
                # For each hexagon, compute gradient of side length w.r.t. its position/angle
                # using finite differences on the fast outer side
                grad_x = np.zeros(N)
                grad_y = np.zeros(N)
                grad_a = np.zeros(N)
                
                eps = 0.001
                for i in range(N):
                    # d(side)/dx_i
                    pos_p = current_pos.copy()
                    pos_p[i, 0] += eps
                    side_p = fast_outer_side(pos_p, current_ang, current_phi_rad)
                    pos_m = current_pos.copy()
                    pos_m[i, 0] -= eps
                    side_m = fast_outer_side(pos_m, current_ang, current_phi_rad)
                    grad_x[i] = (side_p - side_m) / (2 * eps)
                    
                    # d(side)/dy_i
                    pos_p = current_pos.copy()
                    pos_p[i, 1] += eps
                    side_p = fast_outer_side(pos_p, current_ang, current_phi_rad)
                    pos_m = current_pos.copy()
                    pos_m[i, 1] -= eps
                    side_m = fast_outer_side(pos_m, current_ang, current_phi_rad)
                    grad_y[i] = (side_p - side_m) / (2 * eps)
                    
                    # d(side)/dtheta_i
                    ang_p = current_ang.copy()
                    ang_p[i] += eps
                    side_p = fast_outer_side(current_pos, ang_p, current_phi_rad)
                    ang_m = current_ang.copy()
                    ang_m[i] -= eps
                    side_m = fast_outer_side(current_pos, ang_m, current_phi_rad)
                    grad_a[i] = (side_p - side_m) / (2 * eps)
                
                # Compute contact constraints: for each overlapping pair, add repulsive force
                # We use the contact normal to push hexagons apart
                force_x = np.zeros(N)
                force_y = np.zeros(N)
                force_a = np.zeros(N)
                
                for i in range(N):
                    for j in range(i+1, N):
                        pen, n_angle = pairwise_contact(current_pos, current_ang, i, j)
                        if pen > 0:
                            # Push apart along normal
                            fx = math.cos(n_angle)
                            fy = math.sin(n_angle)
                            # i should move in +normal direction, j in -normal
                            force_x[i] += fx * pen
                            force_y[i] += fy * pen
                            force_x[j] -= fx * pen
                            force_y[j] -= fy * pen
                            # Also rotate to reduce penetration
                            # Approximate: rotation that moves the contact point away
                            force_a[i] += 0.1 * pen
                            force_a[j] -= 0.1 * pen
                
                # Combine gradient and contact forces
                # Use projected gradient: move in direction of -grad + contact forces
                for i in range(N):
                    new_x = current_pos[i, 0] - step * grad_x[i] + step * 10 * force_x[i]
                    new_y = current_pos[i, 1] - step * grad_y[i] + step * 10 * force_y[i]
                    new_a = current_ang[i] - step * grad_a[i] + step * 5 * force_a[i]
                    
                    # Check validity
                    test_pos = current_pos.copy()
                    test_ang = current_ang.copy()
                    test_pos[i, 0] = new_x
                    test_pos[i, 1] = new_y
                    test_ang[i] = new_a
                    
                    angles_rad_new = np.radians(test_ang)
                    valid = True
                    h_moved = (test_pos[i, 0], test_pos[i, 1], angles_rad_new[i])
                    for j in range(N):
                        if j == i:
                            continue
                        h_other = (test_pos[j, 0], test_pos[j, 1], angles_rad_new[j])
                        if not hexagons_disjoint_fast(h_moved, h_other):
                            valid = False
                            break
                    
                    if valid:
                        # Evaluate new side
                        new_side = fast_outer_side(test_pos, test_ang, current_phi_rad)
                        if new_side < current_side:
                            current_pos[i, 0] = new_x
                            current_pos[i, 1] = new_y
                            current_ang[i] = new_a
                        else:
                            # Try smaller step
                            for shrink in [0.5, 0.25, 0.1]:
                                sx = current_pos[i, 0] - shrink * step * grad_x[i] + shrink * step * 10 * force_x[i]
                                sy = current_pos[i, 1] - shrink * step * grad_y[i] + shrink * step * 10 * force_y[i]
                                sa = current_ang[i] - shrink * step * grad_a[i] + shrink * step * 5 * force_a[i]
                                
                                test_pos = current_pos.copy()
                                test_ang = current_ang.copy()
                                test_pos[i, 0] = sx
                                test_pos[i, 1] = sy
                                test_ang[i] = sa
                                
                                angles_rad_new = np.radians(test_ang)
                                valid = True
                                h_moved = (test_pos[i, 0], test_pos[i, 1], angles_rad_new[i])
                                for j in range(N):
                                    if j == i:
                                        continue
                                    h_other = (test_pos[j, 0], test_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if valid:
                                    new_side = fast_outer_side(test_pos, test_ang, current_phi_rad)
                                    if new_side < current_side:
                                        current_pos[i, 0] = sx
                                        current_pos[i, 1] = sy
                                        current_ang[i] = sa
                                        break
                
                # Decay step size
                step *= 0.995
                if step < min_step:
                    step = min_step
            
            # Final evaluation
            final_side, final_phi, final_cx, final_cy = compute_min_outer_side(current_pos, current_ang)
            return final_side, current_pos, current_ang, final_phi, final_cx, final_cy
        
        # Run contact network refinement
        cn_time = min(remaining_time * 0.25, 12.0)
        if cn_time > 2.0:
            try:
                cn_side, cn_pos, cn_ang, cn_phi, cn_cx, cn_cy = contact_network_refine(
                    best_positions, best_angles, cn_time)
                if cn_side < best_side - 1e-12:
                    best_side = cn_side
                    best_positions = cn_pos
                    best_angles = cn_ang
                    best_phi = cn_phi
                    best_cx = cn_cx
                    best_cy = cn_cy
            except Exception:
                pass
        
        remaining_time = DEADLINE - (time.time() - t0)
        # 1. Coordinate Descent Polish with tighter final steps
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.04
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
                    step *= 0.6
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.30, 10.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 2. L-BFGS-B on smooth log-sum-exp surrogate with adaptive beta
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 8.0:
            def smooth_outer_side(params, beta=100.0):
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
                            return 1e6 + random.random() * 10
                
                # Compute smooth outer side using log-sum-exp
                all_verts = get_all_vertices(pos, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                offsets = verts_flat - center
                
                # Use current best phi for better initialization
                phi = np.radians(best_phi)
                na = phi + np.pi/6 + np.arange(6) * np.pi/3
                nxx = np.cos(na)
                nyy = np.sin(na)
                pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
                abs_pr = np.abs(pr)
                
                # Log-sum-exp for smooth max
                max_val = np.max(abs_pr, axis=0)
                # Numerically stable log-sum-exp
                shifted = (abs_pr - max_val[:, None]) * beta
                smooth_max = max_val + np.log(np.mean(np.exp(shifted), axis=0)) / beta
                
                return (2.0 / sqrt3) * np.max(smooth_max)

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
            
            # Multi-start with different beta values
            for beta_val in [50.0, 100.0, 200.0]:
                if time.time() - t0 > DEADLINE - 10.0:
                    break
                try:
                    res = minimize(lambda p: smooth_outer_side(p, beta_val), init_params, 
                                   method='L-BFGS-B',
                                   options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
                    
                    if res.fun < best_side + 0.3:
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
                                # Update init_params for next beta
                                for i in range(N):
                                    init_params[3*i] = best_positions[i, 0]
                                    init_params[3*i+1] = best_positions[i, 1]
                                    init_params[3*i+2] = best_angles[i]
                except Exception:
                    pass
        
        # 3. High-precision final polish: coordinate descent with very small steps
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            cd_time2 = min(remaining_time * 0.5, 8.0)
            cd_side2, cd_pos2, cd_ang2 = polish_coordinate_descent(best_positions, best_angles, cd_time2)
            if cd_side2 < best_side:
                best_side = cd_side2
                best_positions = cd_pos2
                best_angles = cd_ang2
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
            
            # 4. Final phi-only refinement with very fine grid
            remaining_time = DEADLINE - (time.time() - t0)
            if remaining_time > 2.0:
                def eval_phi_fine(phi_deg):
                    phi = np.radians(phi_deg)
                    angles_rad = np.radians(best_angles)
                    all_verts = get_all_vertices(best_positions, angles_rad)
                    verts_flat = all_verts.reshape(-1, 2)
                    center = np.mean(verts_flat, axis=0)
                    offsets = verts_flat - center
                    na = phi + np.pi/6 + np.arange(6) * np.pi/3
                    nxx = np.cos(na)
                    nyy = np.sin(na)
                    pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
                    return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()
                
                # Multi-start fine grid search for phi with tighter bounds
                best_phi_fine = best_phi
                best_side_fine = eval_phi_fine(best_phi)
                
                # First do a coarse 0.1-degree grid over full range to catch global min
                phi_coarse = np.arange(0, 60, 0.1)
                for chunk_start in range(0, len(phi_coarse), 500):
                    chunk = phi_coarse[chunk_start:chunk_start+500]
                    sides_chunk = np.array([eval_phi_fine(p) for p in chunk])
                    min_idx = np.argmin(sides_chunk)
                    if sides_chunk[min_idx] < best_side_fine:
                        best_side_fine = sides_chunk[min_idx]
                        best_phi_fine = float(chunk[min_idx])
                
                # Then refine locally around best found
                for start_phi in [best_phi_fine - 0.5, best_phi_fine, best_phi_fine + 0.5]:
                    lo = max(0.0, start_phi - 0.3)
                    hi = min(60.0, start_phi + 0.3)
                    try:
                        res = minimize_scalar(eval_phi_fine, bounds=(lo, hi), method='bounded',
                                              options={'xatol': 1e-14, 'maxiter': 500})
                        if res.success and res.fun < best_side_fine:
                            best_side_fine = res.fun
                            best_phi_fine = res.x
                    except Exception:
                        pass
                
                if best_side_fine < best_side - 1e-12:
                    best_side = best_side_fine
                    best_phi = best_phi_fine
                    _, _, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        # Fall back to best valid initial config
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # One last check: try to tighten phi with exact computation
    try:
        angles_rad_final = np.radians(best_angles)
        all_verts_final = get_all_vertices(best_positions, angles_rad_final)
        verts_flat_final = all_verts_final.reshape(-1, 2)
        center_final = np.mean(verts_flat_final, axis=0)
        offsets_final = verts_flat_final - center_final
        
        def exact_side_for_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets_final[:, 0:1] * nxx[None, :] + offsets_final[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()
        
        # Fine grid around current best
        phi_grid = np.linspace(max(0, best_phi - 0.5), min(60, best_phi + 0.5), 101)
        sides_grid = np.array([exact_side_for_phi(p) for p in phi_grid])
        best_idx = np.argmin(sides_grid)
        if sides_grid[best_idx] < best_side - 1e-12:
            best_phi = float(phi_grid[best_idx])
            best_side = float(sides_grid[best_idx])
            best_cx = float(center_final[0])
            best_cy = float(center_final[1])
    except Exception:
        pass
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
