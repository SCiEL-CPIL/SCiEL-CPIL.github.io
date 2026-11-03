import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

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
    sqrt3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        rx = _UNIT_HEX_VERTS[:, 0] * c - _UNIT_HEX_VERTS[:, 1] * s
        ry = _UNIT_HEX_VERTS[:, 0] * s + _UNIT_HEX_VERTS[:, 1] * c
        return np.column_stack((rx + cx, ry + cy))

    def get_hex_normals(angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        rx = _EDGE_NORMALS[:, 0] * c - _EDGE_NORMALS[:, 1] * s
        ry = _EDGE_NORMALS[:, 0] * s + _EDGE_NORMALS[:, 1] * c
        return np.column_stack((rx, ry))

    def check_overlap(v1, n1, v2, n2):
        for i in range(6):
            nx, ny = n1[i]
            proj1 = v1[:, 0] * nx + v1[:, 1] * ny
            proj2 = v2[:, 0] * nx + v2[:, 1] * ny
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
        for i in range(6):
            nx, ny = n2[i]
            proj1 = v1[:, 0] * nx + v1[:, 1] * ny
            proj2 = v2[:, 0] * nx + v2[:, 1] * ny
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
        return False

    def compute_min_side_general(positions, angles_deg, phi_outer_deg=0.0):
        # positions: (N, 2), angles_deg: (N,)
        all_verts = []
        for i in range(N):
            v = get_hex_vertices(positions[i, 0], positions[i, 1], math.radians(angles_deg[i]))
            all_verts.append(v)
        verts_flat = np.vstack(all_verts)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi = math.radians(phi_outer_deg)
        normal_angles = phi + math.pi/6 + np.arange(6) * math.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        return np.max(side_vals)

    def objective_general(params):
        # params: 3*N + 1 (x, y, a for each, plus phi_outer)
        if time.time() - t0 > DEADLINE:
            return 10000
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles[i] = params[3*i+2]
        phi = params[3*N]
        
        # Check overlaps
        for i in range(N):
            v1 = get_hex_vertices(positions[i, 0], positions[i, 1], math.radians(angles[i]))
            n1 = get_hex_normals(math.radians(angles[i]))
            for j in range(i+1, N):
                v2 = get_hex_vertices(positions[j, 0], positions[j, 1], math.radians(angles[j]))
                n2 = get_hex_normals(math.radians(angles[j]))
                if not check_overlap(v1, n1, v2, n2):
                    return 1000.0
        
        side = compute_min_side_general(positions, angles, phi)
        return side

    # 1. Symmetric Optimization
    def objective_symmetric(params):
        if time.time() - t0 > DEADLINE:
            return 10000
        x1, y1, a1_deg, x2, y2, a2_deg, phi_outer_deg = params
        a1 = math.radians(a1_deg)
        a2 = math.radians(a2_deg)
        phi = math.radians(phi_outer_deg)
        
        v1 = get_hex_vertices(x1, y1, a1)
        n1 = get_hex_normals(a1)
        v2 = get_hex_vertices(x2, y2, a2)
        n2 = get_hex_normals(a2)
        
        if not check_overlap(v1, n1, v2, n2): return 1000.0
        
        c60, s60 = math.cos(math.pi/3), math.sin(math.pi/3)
        x1p = x1*c60 - y1*s60; y1p = x1*s60 + y1*c60
        v1p = get_hex_vertices(x1p, y1p, a1 + math.pi/3)
        n1p = get_hex_normals(a1 + math.pi/3)
        if not check_overlap(v1, n1, v1p, n1p): return 1000.0
        
        x2p = x2*c60 - y2*s60; y2p = x2*s60 + y2*c60
        v2p = get_hex_vertices(x2p, y2p, a2 + math.pi/3)
        n2p = get_hex_normals(a2 + math.pi/3)
        if not check_overlap(v2, n2, v2p, n2p): return 1000.0
        
        if not check_overlap(v1, n1, v2p, n2p): return 1000.0
        if not check_overlap(v2, n2, v1p, n1p): return 1000.0
        
        # Compute side
        all_verts = []
        for k in range(6):
            rot = k * math.pi / 3
            cr, sr = math.cos(rot), math.sin(rot)
            x1r = x1*cr - y1*sr; y1r = x1*sr + y1*cr
            v1r = get_hex_vertices(x1r, y1r, a1 + rot)
            all_verts.append(v1r)
            x2r = x2*cr - y2*sr; y2r = x2*sr + y2*cr
            v2r = get_hex_vertices(x2r, y2r, a2 + rot)
            all_verts.append(v2r)
            
        verts_flat = np.vstack(all_verts)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        normal_angles = phi + math.pi/6 + np.arange(6) * math.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        return np.max(side_vals)

    best_side = float('inf')
    best_params_general = None
    
    # Run Symmetric
    inits_sym = [
        [2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0],
        [2.5, 1.0, 30.0, 1.0, 2.0, 30.0, 0.0],
        [4.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
        [1.7, 0.0, 0.0, 1.7, 1.0, 0.0, 0.0]
    ]
    
    for init in inits_sym:
        try:
            res = minimize(objective_symmetric, init, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_side and res.fun < 100:
                best_side = res.fun
                # Convert symmetric to general params for refinement
                x1, y1, a1, x2, y2, a2, phi = res.x
                params_gen = np.zeros(3*N + 1)
                for k in range(6):
                    rot = k * math.pi / 3
                    cr, sr = math.cos(rot), math.sin(rot)
                    x1r = x1*cr - y1*sr; y1r = x1*sr + y1*cr
                    a1r = (a1 + rot) * 180 / math.pi
                    params_gen[3*k] = x1r
                    params_gen[3*k+1] = y1r
                    params_gen[3*k+2] = a1r
                    
                    x2r = x2*cr - y2*sr; y2r = x2*sr + y2*cr
                    a2r = (a2 + rot) * 180 / math.pi
                    params_gen[3*k+3] = x2r
                    params_gen[3*k+4] = y2r
                    params_gen[3*k+5] = a2r
                params_gen[3*N] = phi
                best_params_general = params_gen
        except Exception:
            pass

    # Run General Refinement if time permits
    if best_params_general is not None and time.time() - t0 < DEADLINE - 10:
        try:
            res_gen = minimize(objective_general, best_params_general, method='Powell',
                               options={'maxiter': 200, 'ftol': 1e-8})
            if res_gen.fun < best_side:
                best_side = res_gen.fun
                best_params_general = res_gen.x
        except Exception:
            pass

    # Fallback if no good solution
    if best_params_general is None:
        # Honeycomb
        a1_vec = np.array([sqrt3, 0.0])
        a2_vec = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0)
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1_vec + j * a2_vec
        params_gen = np.zeros(3*N + 1)
        for i in range(N):
            params_gen[3*i] = positions[i,0]
            params_gen[3*i+1] = positions[i,1]
            params_gen[3*i+2] = angles[i]
        params_gen[3*N] = 0.0
        best_params_general = params_gen
        best_side = objective_general(params_gen)

    # Extract result
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        positions[i, 0] = best_params_general[3*i]
        positions[i, 1] = best_params_general[3*i+1]
        angles[i] = best_params_general[3*i+2]
    phi = best_params_general[3*N]
    
    # Compute center
    all_verts = []
    for i in range(N):
        v = get_hex_vertices(positions[i, 0], positions[i, 1], math.radians(angles[i]))
        all_verts.append(v)
    verts_flat = np.vstack(all_verts)
    center = np.mean(verts_flat, axis=0)
    
    inner = [[float(positions[i,0]), float(positions[i,1]), float(angles[i])] for i in range(N)]
    
    return inner, (float(center[0]), float(center[1])), float(best_side), float(phi)
