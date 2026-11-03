import math
import time
import random
import numpy as np
from scipy.optimize import minimize

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

    def smooth_max(x, beta):
        """Vectorized soft-max approximation of max."""
        # x shape: (..., M)
        mx = np.max(x, axis=-1, keepdims=True)
        # Prevent overflow
        shifted = beta * (x - mx)
        # Clip to avoid exp overflow in log-sum-exp
        shifted = np.clip(shifted, -50, 50)
        sum_exp = np.sum(np.exp(shifted), axis=-1, keepdims=True)
        return (mx + np.log(sum_exp) / beta).squeeze(-1)

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def compute_smooth_outer_side(positions, angles_rad, outer_angle_rad, beta=50.0):
        """
        Computes a smooth approximation of the minimum outer hexagon side length.
        """
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Estimate center as midpoint of bounding box for stability
        center = (verts_flat.max(axis=0) + verts_flat.min(axis=0)) / 2.0
        offsets = verts_flat - center
        
        # Outer hexagon normals depend on outer_angle_rad
        # Normals are at outer_angle + pi/6 + k*pi/3
        normal_angles = outer_angle_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # Project all vertices onto all 6 normals
        # offsets: (72, 2), nx: (6,) -> proj: (72, 6)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        
        # For each normal direction (edge), find the max absolute projection
        # This gives the "radius" in that direction
        # We need the max of these 6 radii to determine the side length
        # side = (2/sqrt(3)) * max_radius
        
        # Use soft-max to smooth the max over the 6 edges
        max_radii = smooth_max(abs_proj, beta) # Shape: (72,) ? No.
        
        # Wait, smooth_max reduces the last axis.
        # abs_proj is (72, 6). We want max over the 6 edges for EACH vertex? No.
        # The condition is: For a fixed outer hexagon, the side length is determined by the vertex that is furthest out.
        # side >= (2/sqrt(3)) * |proj| for all vertices and all edges.
        # So side = (2/sqrt(3)) * max_{v, e} |proj(v, e)|
        
        # We want to smooth the max over ALL vertices and ALL edges.
        # Flatten abs_proj to (72*6,)
        flat_proj = abs_proj.flatten()
        
        # Soft max over all 432 values
        max_abs_proj = smooth_max(flat_proj, beta)
        
        side = (2.0 / sqrt3) * max_abs_proj
        return side

    def compute_smooth_overlap_penalty(positions, angles_rad, beta=50.0):
        """
        Computes a smooth penalty for overlapping inner hexagons.
        Uses soft-min of separations along normals.
        """
        penalty = 0.0
        n_hex = len(positions)
        
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                # Check distance first for speed
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                # Let's use a larger cutoff to be safe, say 3.0 squared = 9.0.
                if dist_sq > 9.0:
                    continue
                
                a1 = angles_rad[i]
                a2 = angles_rad[j]
                
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)
                
                # Vertices
                v1x = positions[i, 0] + _ux * c1 - _uy * s1
                v1y = positions[i, 1] + _ux * s1 + _uy * c1
                v2x = positions[j, 0] + _ux * c2 - _uy * s2
                v2y = positions[j, 1] + _ux * s2 + _uy * c2
                
                # Normals for hex i
                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                # Normals for hex j
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2
                
                separations = []
                
                # Check axes from hex i
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    # Separation: min1 - max2 (if positive, disjoint)
                    sep = p1.min() - p2.max()
                    separations.append(sep)
                
                # Check axes from hex j
                for k in range(6):
                    nx, ny = en2x[k], en2y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    sep = p1.min() - p2.max()
                    separations.append(sep)
                
                seps = np.array(separations)
                
                # We want to penalize if any separation is negative.
                # Let u = -seps. If all seps > 0, all u < 0.
                # We want a smooth function that is 0 when max(u) < 0 (i.e., min(sep) > 0).
                # Use soft_max on u.
                # If max(u) is negative, soft_max is negative.
                # We can clamp: penalty_part = smooth_max(u, beta)
                # If penalty_part > 0, there is overlap.
                # To make it smooth and zero when disjoint, we can use:
                # r = smooth_max(u, beta)
                # penalty += r * r (if r > 0)
                # But r is smooth. r^2 is smooth.
                # However, if r is slightly negative (disjoint but close), r^2 is small but positive.
                # This creates a "repulsive" force which is good for optimization.
                
                r = smooth_max(-seps, beta)
                penalty += r * r
        
        return penalty

    def objective(params):
        """
        Combined smooth objective function.
        params: [x0, y0, a0, x1, y1, a1, ...]
        """
        if time.time() - t0 > DEADLINE:
            return 1e10
            
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        # Outer angle is also a variable? 
        # The task says "outer hexagon center (x, y), side length and rotation".
        # We can optimize the outer angle as well to find the best fit.
        # But the outer center is determined by the inner positions.
        # So we only need to optimize the outer angle.
        # Let's add outer_angle as the last parameter.
        
        # Wait, the current signature returns center, side, angle.
        # If we optimize outer_angle, we can find a tighter fit.
        
        # Let's assume the outer angle is a variable in params.
        # params length: 3*N + 1
        outer_angle = params[3*N]
        
        angles_rad = np.radians(ang)
        outer_angle_rad = np.radians(outer_angle)
        
        side = compute_smooth_outer_side(pos, angles_rad, outer_angle_rad, beta=50.0)
        overlap = compute_smooth_overlap_penalty(pos, angles_rad, beta=50.0)
        
        # Penalty weight
        # If overlap is large, side is irrelevant.
        # We want to minimize side, but only if overlap is 0.
        # Use a large penalty for overlap.
        penalty_weight = 100.0
        total = side + penalty_weight * overlap
        
        return total

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
    best_outer_angle = 0.0
    best_cx = 0.0
    best_cy = 0.0

    def evaluate_exact(positions, angles_deg, outer_angle_deg):
        """Exact evaluation for final result."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = (verts_flat.max(axis=0) + verts_flat.min(axis=0)) / 2.0
        offsets = verts_flat - center
        
        phi = np.radians(outer_angle_deg)
        normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side = np.max(side_vals)
        
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                # Reuse fast disjoint check
                if not hexagons_disjoint_fast(h1, h2):
                    valid = False
                    break
            if not valid: break
        
        if not valid:
            return float('inf'), 0.0, 0.0, 0.0
            
        return float(side), float(outer_angle_deg), float(center[0]), float(center[1])

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

    # Optimization Strategy
    # 1. Initialize with honeycomb
    # 2. Use L-BFGS-B to minimize the smooth objective
    # 3. Restart with different initial configurations if time permits
    
    def run_optimization(init_pos, init_ang, init_outer_angle, time_limit):
        init_params = np.zeros(3*N + 1)
        for i in range(N):
            init_params[3*i] = init_pos[i, 0]
            init_params[3*i+1] = init_pos[i, 1]
            init_params[3*i+2] = init_ang[i]
        init_params[3*N] = init_outer_angle
        
        t_start = time.time()
        
        try:
            # L-BFGS-B
            res = minimize(objective, init_params, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000})
            
            if res.fun < 1e5:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                new_outer_angle = res.x[3*N]
                
                # Evaluate exact
                side, ang, cx, cy = evaluate_exact(new_pos, new_ang, new_outer_angle)
                return side, new_pos, new_ang, new_outer_angle, cx, cy
        except Exception:
            pass
            
        return float('inf'), None, None, None, 0.0, 0.0

    # Multiple restarts
    initial_configs = []
    # 1. Honeycomb
    initial_configs.append((base_positions.copy(), base_angles.copy(), 0.0))
    
    # 2. Honeycomb rotated 30 deg
    p2 = base_positions.copy()
    a2 = base_angles.copy() + 30.0
    initial_configs.append((p2, a2, 30.0))
    
    # 3. Scaled honeycomb (0.95)
    p3 = base_positions * 0.95
    a3 = base_angles.copy()
    initial_configs.append((p3, a3, 0.0))
    
    # 4. Ring
    p4 = np.zeros((N, 2))
    a4 = np.zeros(N)
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        p4[i, 0] = r * np.cos(theta)
        p4[i, 1] = r * np.sin(theta)
        a4[i] = theta * 180 / np.pi
    initial_configs.append((p4, a4, 0.0))
    
    # Run optimization for each config
    time_per_restart = (DEADLINE - (time.time() - t0)) / len(initial_configs)
    
    for i, (pos, ang, o_ang) in enumerate(initial_configs):
        if time.time() - t0 > DEADLINE:
            break
            
        t_start = time.time()
        time_limit = min(time_per_restart, DEADLINE - (time.time() - t0))
        
        side, new_pos, new_ang, new_o_ang, cx, cy = run_optimization(pos, ang, o_ang, time_limit)
        
        if side < best_side:
            best_side = side
            best_positions = new_pos
            best_angles = new_ang
            best_outer_angle = new_o_ang
            best_cx = cx
            best_cy = cy

    # Final Polish with Nelder-Mead on exact objective if time remains
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0 and best_positions is not None:
        init_params = np.zeros(3*N + 1)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
        init_params[3*N] = best_outer_angle
        
        def exact_objective(params):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            o_ang = params[3*N]
            side, _, _, _ = evaluate_exact(pos, ang, o_ang)
            return side
        
        try:
            res = minimize(exact_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9})
            
            if res.fun < best_side:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                new_o_ang = res.x[3*N]
                
                side, _, cx, cy = evaluate_exact(new_pos, new_ang, new_o_ang)
                if side < best_side:
                    best_side = side
                    best_positions = new_pos
                    best_angles = new_ang
                    best_outer_angle = new_o_ang
                    best_cx = cx
                    best_cy = cy
        except Exception:
            pass

    # Final verification and result construction
    if best_positions is None:
        # Fallback to initial
        best_positions = base_positions
        best_angles = base_angles
        best_outer_angle = 0.0
        best_side, best_outer_angle, best_cx, best_cy = evaluate_exact(best_positions, best_angles, best_outer_angle)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_outer_angle)
    
    return inner, center, side, angle
