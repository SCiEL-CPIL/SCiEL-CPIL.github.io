import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 115.0  # Safety margin

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
    
    # Precomputed constants
    sqrt3 = math.sqrt(3)
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def compute_overlap_penalty(positions, angles_rad):
        """
        Returns a penalty value based on the squared penetration depth of overlaps.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Quick check: if centers are far apart, no overlap
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.5: # Max distance between centers for contact is 2*radius = 2, squared 4. 4.5 is safe margin
                    continue
                
                a1 = angles_rad[i]
                a2 = angles_rad[j]
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)
                
                v1x = positions[i, 0] + _ux * c1 - _uy * s1
                v1y = positions[i, 1] + _ux * s1 + _uy * c1
                v2x = positions[j, 0] + _ux * c2 - _uy * s2
                v2y = positions[j, 1] + _ux * s2 + _uy * c2
                
                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2
                
                max_penetration = 0.0
                # Check separating axes for hex i
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1_min = np.min(v1x * nx + v1y * ny)
                    p1_max = np.max(v1x * nx + v1y * ny)
                    p2_min = np.min(v2x * nx + v2y * ny)
                    p2_max = np.max(v2x * nx + v2y * ny)
                    
                    if p1_max < p2_min or p2_max < p1_min:
                        # Separated, no overlap
                        max_penetration = 0.0
                        break
                    else:
                        # Overlap on this axis
                        pen = min(p1_max, p2_max) - max(p1_min, p2_min)
                        if pen > max_penetration:
                            max_penetration = pen
                
                # If we broke early, no overlap. Otherwise, check hex j axes too?
                # For penalty, if they overlap, the max penetration on ANY axis is a measure of depth.
                # But we need to ensure they actually overlap.
                # If max_penetration > 0, they overlap.
                # To be robust, we should check if they are separated on ALL axes.
                # If they are separated on any axis, penalty is 0.
                # If not separated on any axis, penalty is max penetration.
                
                separated = False
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1_min = np.min(v1x * nx + v1y * ny)
                    p1_max = np.max(v1x * nx + v1y * ny)
                    p2_min = np.min(v2x * nx + v2y * ny)
                    p2_max = np.max(v2x * nx + v2y * ny)
                    if p1_max < p2_min or p2_max < p1_min:
                        separated = True
                        break
                if not separated:
                    for k in range(6):
                        nx, ny = en2x[k], en2y[k]
                        p1_min = np.min(v1x * nx + v1y * ny)
                        p1_max = np.max(v1x * nx + v1y * ny)
                        p2_min = np.min(v2x * nx + v2y * ny)
                        p2_max = np.max(v2x * nx + v2y * ny)
                        if p1_max < p2_min or p2_max < p1_min:
                            separated = True
                            break
                
                if not separated:
                    # Calculate max penetration depth
                    # We already have max_penetration from the first loop if it didn't break?
                    # The first loop breaks if separated. If it doesn't break, it accumulates max_penetration.
                    # But if it breaks, max_penetration is 0.
                    # So if not separated, max_penetration holds the value.
                    penalty += max_penetration * max_penetration

        return penalty

    def compute_side_length(positions, angles_rad, phi_rad):
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        normal_angles = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        return np.max(side_vals)

    def objective(params):
        # params: [x1, y1, a1, ..., x12, y12, a12, phi]
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
        phi_deg = params[-1]
        
        angles_rad = np.radians(angles_deg)
        phi_rad = np.radians(phi_deg)
        
        overlap = compute_overlap_penalty(positions, angles_rad)
        side = compute_side_length(positions, angles_rad, phi_rad)
        
        # Penalty weight
        # If overlap is non-zero, it should dominate
        return 1000.0 * overlap + side

    # Initial Honeycomb Configuration
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
    best_params = None

    # Multi-start Nelder-Mead
    random.seed(42)
    start_time = time.time()
    
    # We will run several optimizations
    # Start 1: Honeycomb
    # Start 2-5: Random perturbations of Honeycomb
    
    starts = []
    for idx, (i, j) in enumerate(lattice_coords):
        starts.append(base_positions[idx].copy())
    
    # Create initial parameter vector for honeycomb
    init_params = np.zeros(3*N + 1)
    for i in range(N):
        init_params[3*i] = base_positions[i, 0]
        init_params[3*i+1] = base_positions[i, 1]
        init_params[3*i+2] = base_angles[i]
    init_params[-1] = 0.0 # phi
    starts_params = [init_params]
    
    # Add some random starts
    for _ in range(4):
        p = init_params.copy()
        for i in range(N):
            p[3*i] += random.gauss(0, 0.5)
            p[3*i+1] += random.gauss(0, 0.5)
            p[3*i+2] += random.gauss(0, 10.0)
        p[-1] = random.uniform(0, 60)
        starts_params.append(p)

    for start_p in starts_params:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        try:
            res = minimize(objective, start_p, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6})
            
            if res.fun < best_side:
                # Extract best side length
                positions = np.zeros((N, 2))
                angles_deg = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = res.x[3*i]
                    positions[i, 1] = res.x[3*i+1]
                    angles_deg[i] = res.x[3*i+2]
                phi_deg = res.x[-1]
                
                angles_rad = np.radians(angles_deg)
                phi_rad = np.radians(phi_deg)
                
                # Verify validity
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        if dx*dx + dy*dy <= 4.5:
                            # Detailed check
                            a1 = angles_rad[i]
                            a2 = angles_rad[j]
                            c1, s1 = math.cos(a1), math.sin(a1)
                            c2, s2 = math.cos(a2), math.sin(a2)
                            v1x = positions[i, 0] + _ux * c1 - _uy * s1
                            v1y = positions[i, 1] + _ux * s1 + _uy * c1
                            v2x = positions[j, 0] + _ux * c2 - _uy * s2
                            v2y = positions[j, 1] + _ux * s2 + _uy * c2
                            en1x = _enx * c1 - _eny * s1
                            en1y = _enx * s1 + _eny * c1
                            en2x = _enx * c2 - _eny * s2
                            en2y = _enx * s2 + _eny * c2
                            
                            sep = False
                            for k in range(6):
                                nx, ny = en1x[k], en1y[k]
                                p1_min = np.min(v1x * nx + v1y * ny)
                                p1_max = np.max(v1x * nx + v1y * ny)
                                p2_min = np.min(v2x * nx + v2y * ny)
                                p2_max = np.max(v2x * nx + v2y * ny)
                                if p1_max < p2_min or p2_max < p1_min:
                                    sep = True
                                    break
                            if not sep:
                                for k in range(6):
                                    nx, ny = en2x[k], en2y[k]
                                    p1_min = np.min(v1x * nx + v1y * ny)
                                    p1_max = np.max(v1x * nx + v1y * ny)
                                    p2_min = np.min(v2x * nx + v2y * ny)
                                    p2_max = np.max(v2x * nx + v2y * ny)
                                    if p1_max < p2_min or p2_max < p1_min:
                                        sep = True
                                        break
                            if not sep:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    side = compute_side_length(positions, angles_rad, phi_rad)
                    if side < best_side:
                        best_side = side
                        best_params = res.x.copy()
        except Exception:
            pass

    # Fallback to honeycomb if optimization failed or didn't improve
    if best_params is None:
        best_params = init_params.copy()
        best_side = compute_side_length(base_positions, np.zeros(N), 0.0)

    # Final extraction
    positions = np.zeros((N, 2))
    angles_deg = np.zeros(N)
    for i in range(N):
        positions[i, 0] = best_params[3*i]
        positions[i, 1] = best_params[3*i+1]
        angles_deg[i] = best_params[3*i+2]
    phi_deg = best_params[-1]
    
    angles_rad = np.radians(angles_deg)
    phi_rad = np.radians(phi_deg)
    
    # Recalculate center and side for final output
    all_verts = get_all_vertices(positions, angles_rad)
    verts_flat = all_verts.reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    side = compute_side_length(positions, angles_rad, phi_rad)
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles_deg[i])])
        
    center_out = (float(center[0]), float(center[1]))
    side_out = float(side)
    angle_out = float(phi_deg)
    
    return inner, center_out, side_out, angle_out
