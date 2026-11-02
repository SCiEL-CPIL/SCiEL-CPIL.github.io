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
    
    # Helper: Get vertices of all hexagons given positions and angles
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        return all_verts

    def hexagons_disjoint_fast(h1, h2):
        """
        Fast SAT check for two unit hexagons.
        h1, h2: (x, y, angle_rad)
        Uses precomputed edge normals rotated by each hex's angle.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # Conservative early exit: max center distance for any contact is 2.0
        # (vertex-to-vertex of two unit hexagons). Use 4.5 as squared bound.
        if dist_sq > 4.5:
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Compute vertices
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        
        # Rotate edge normals for each hex
        en = edge_normals  # (6, 2)
        en1x = en[:, 0] * c1 - en[:, 1] * s1
        en1y = en[:, 0] * s1 + en[:, 1] * c1
        en2x = en[:, 0] * c2 - en[:, 1] * s2
        en2y = en[:, 0] * s2 + en[:, 1] * c2
        
        # Check all 12 normals
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
        
        # Vectorized evaluation for a batch of phi values
        def eval_phi_batch(phi_vals):
            """phi_vals: array of degrees. Returns array of side lengths."""
            phi_rad = np.radians(phi_vals)  # (M,)
            # 6 normal directions for each phi: (M, 6)
            normal_angles = phi_rad[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)  # (M, 6)
            ny = np.sin(normal_angles)  # (M, 6)
            
            # Project all vertices onto each normal: (M, 72, 6)
            proj = offsets[None, :, 0:1] * nx[:, None, :] + offsets[None, :, 1:2] * ny[:, None, :]
            abs_proj = np.abs(proj)  # (M, 72, 6)
            
            # For each (phi, direction), get max absolute projection
            max_projs = np.max(abs_proj, axis=1)  # (M, 6)
            side_vals = (2.0 / sqrt3) * max_projs  # (M, 6)
            return np.max(side_vals, axis=1)  # (M,)
        
        # Coarse scan: 60 points
        coarse_phis = np.linspace(0, 60, 61, endpoint=False)
        coarse_sides = eval_phi_batch(coarse_phis)
        coarse_idx = np.argmin(coarse_sides)
        
        # Fine scan around best coarse point
        best_coarse_phi = coarse_phis[coarse_idx]
        # Handle wraparound
        fine_phis = np.linspace(best_coarse_phi - 1.0, best_coarse_phi + 1.0, 41)
        # Clamp to [0, 60)
        fine_phis = fine_phis % 60.0
        fine_sides = eval_phi_batch(fine_phis)
        fine_idx = np.argmin(fine_sides)
        
        # Ultra-fine scan around best fine point
        best_fine_phi = fine_phis[fine_idx]
        ultra_phis = np.linspace(best_fine_phi - 0.05, best_fine_phi + 0.05, 21)
        ultra_phis = ultra_phis % 60.0
        ultra_sides = eval_phi_batch(ultra_phis)
        ultra_idx = np.argmin(ultra_sides)
        
        best_phi = ultra_phis[ultra_idx]
        best_side = ultra_sides[ultra_idx]
        
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
    num_restarts = 3
    time_per_restart = (DEADLINE * 0.8) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE * 0.8:
            break
            
        # Perturb initial configuration
        current_positions = base_positions.copy()
        current_angles = base_angles.copy()
        if r > 0:
            current_positions += np.random.normal(0, 0.2, size=(N, 2))
            current_angles += np.random.normal(0, 10.0, size=N)
            
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        
        if current_side < best_side:
            best_side = current_side
            best_positions = current_positions.copy()
            best_angles = current_angles.copy()
            best_phi = current_phi
            best_cx = current_cx
            best_cy = current_cy

        T = 0.4
        T_min = 0.0001
        cooling = 0.9998
        step_size = 0.25
        step_decay = 0.99995
        
        step_count = 0
        max_steps = 50000 # Adjust per restart
        
        while time.time() - t0 < DEADLINE * 0.8 and step_count < max_steps:
            step_count += 1
            
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
            if step_size < 0.01: step_size = 0.01

    # Polish Step: coordinate-descent style local optimization
    def polish_coordinate_descent(pos, ang, max_iter=2000):
        """Local polish using coordinate descent on each hex's position and angle."""
        current_side, current_phi, cx, cy = compute_min_outer_side(pos, ang)
        best_side_local = current_side
        best_pos_local = pos.copy()
        best_ang_local = ang.copy()
        best_phi_local = current_phi
        
        step_pos = 0.05
        step_ang = 1.0
        no_improve = 0
        max_no_improve = 100
        
        for it in range(max_iter):
            if time.time() - t0 > DEADLINE - 3.0:
                break
            if no_improve > max_no_improve:
                break
            
            improved = False
            for i in range(N):
                # Try small perturbations in x, y, angle
                for dx in [step_pos, -step_pos, 0]:
                    for dy in [step_pos, -step_pos, 0]:
                        for da in [step_ang, -step_ang, 0]:
                            if dx == 0 and dy == 0 and da == 0:
                                continue
                            test_pos = pos.copy()
                            test_ang = ang.copy()
                            test_pos[i, 0] += dx
                            test_pos[i, 1] += dy
                            test_ang[i] += da
                            
                            # Quick validity check for moved hex
                            angles_rad_test = np.radians(test_ang)
                            valid = True
                            h_moved = (test_pos[i, 0], test_pos[i, 1], angles_rad_test[i])
                            for j in range(N):
                                if j == i: continue
                                h_other = (test_pos[j, 0], test_pos[j, 1], angles_rad_test[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            side_test, phi_test, cx_test, cy_test = compute_min_outer_side(test_pos, test_ang)
                            if side_test < best_side_local - 1e-9:
                                best_side_local = side_test
                                best_pos_local = test_pos.copy()
                                best_ang_local = test_ang.copy()
                                best_phi_local = phi_test
                                pos = test_pos.copy()
                                ang = test_ang.copy()
                                improved = True
                                no_improve = 0
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                # Reduce step size
                step_pos *= 0.7
                step_ang *= 0.7
                if step_pos < 0.001:
                    break
        
        return best_pos_local, best_ang_local, best_side_local, best_phi_local, cx, cy

    # Run coordinate descent polish
    if time.time() - t0 < DEADLINE - 5.0:
        best_positions, best_angles, best_side, best_phi, best_cx, best_cy = \
            polish_coordinate_descent(best_positions, best_angles, max_iter=3000)
        
        # Second pass with tighter steps
        if time.time() - t0 < DEADLINE - 3.0:
            best_positions, best_angles, best_side, best_phi, best_cx, best_cy = \
                polish_coordinate_descent(best_positions, best_angles, max_iter=2000)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
