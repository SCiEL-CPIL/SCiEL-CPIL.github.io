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
    
    # Precompute unit hexagon vertices (angle 0)
    # Vertices at 0, 60, 120, 180, 240, 360 degrees
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
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
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If far apart, disjoint. Max distance between centers of two unit hexagons
        # that can still touch is 2*apothem*2 = 2*sqrt(3) ≈ 3.464
        if dist_sq > 12.0: # dist > 3.464
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        v1 = np.column_stack((v1x, v1y))
        
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        v2 = np.column_stack((v2x, v2y))
        
        # SAT
        # Normals for v1
        for i in range(6):
            ex = v1[(i+1)%6, 0] - v1[i, 0]
            ey = v1[(i+1)%6, 1] - v1[i, 1]
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9: continue
            nx /= norm
            ny /= norm
            
            proj1 = v1[:, 0]*nx + v1[:, 1]*ny
            proj2 = v2[:, 0]*nx + v2[:, 1]*ny
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
                
        for i in range(6):
            ex = v2[(i+1)%6, 0] - v2[i, 0]
            ey = v2[(i+1)%6, 1] - v2[i, 1]
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9: continue
            nx /= norm
            ny /= norm
            
            proj1 = v1[:, 0]*nx + v1[:, 1]*ny
            proj2 = v2[:, 0]*nx + v2[:, 1]*ny
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
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
        best_side = float('inf')
        best_phi = 0.0
        
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
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
                
        # Fine scan
        for phi_deg in np.arange(best_phi - 2, best_phi + 2, 0.05):
            if phi_deg < 0 or phi_deg >= 60: continue
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

    # Initial Configuration: Honeycomb-like
    # 3x4 grid
    cols = 4
    rows = 3
    x0 = -(cols - 1) / 2.0 * 3.0
    y0 = -(rows - 1) / 2.0 * math.sqrt(3)
    
    initial_positions = np.zeros((N, 2))
    initial_angles = np.zeros(N)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            initial_positions[idx, 0] = x0 + c * 3.0
            initial_positions[idx, 1] = y0 + r * math.sqrt(3)
            initial_angles[idx] = 0.0
            idx += 1
            
    best_side = float('inf')
    best_positions = initial_positions.copy()
    best_angles = initial_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    current_positions = initial_positions.copy()
    current_angles = initial_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    step_size = 0.5
    
    step_count = 0
    max_steps = 50000
    
    # SA Loop
    while time.time() - t0 < DEADLINE * 0.8 and step_count < max_steps:
        step_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 10.0) # degrees
        
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

    # Polish Step
    # Use Nelder-Mead on the best configuration found
    # Variables: 36 (x, y, angle for each hex)
    def polish_objective(params):
        # Unpack
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        # Check validity
        angles_rad = np.radians(ang)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return 1e6 # Penalty
        
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        return side

    # Initial params for polish
    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = best_positions[i, 0]
        init_params[3*i+1] = best_positions[i, 1]
        init_params[3*i+2] = best_angles[i]
        
    # Run minimize
    if time.time() - t0 < DEADLINE * 0.95:
        try:
            res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.fun < best_side:
                # Update best
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                    
                # Verify validity one last time
                if is_valid_configuration(new_pos, new_ang):
                    best_side = res.fun
                    best_positions = new_pos
                    best_angles = new_ang
                    # Recompute phi and center for final output
                    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        except Exception:
            pass # Ignore errors in polish, keep SA result

    # Final Result
    # Construct inner list: (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
