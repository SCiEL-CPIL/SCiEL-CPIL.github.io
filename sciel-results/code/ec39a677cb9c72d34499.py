import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0

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
    
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
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
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def objective_function(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        angles_rad = np.radians(ang)
        
        # Check for overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Calculate minimum separation violation roughly
                    # For simplicity, just add a large penalty
                    overlap_penalty += 100.0
                    
        if overlap_penalty > 0:
            return 1000.0 + overlap_penalty

        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        return side

    # 1. Generate Initial Configuration: Honeycomb Cluster
    # Lattice vectors
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    
    # Coordinates for 12 hexagons in a compact cluster
    # (0,0) + 6 neighbors + 5 from next ring
    lattice_coords = [
        (0, 0), 
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (1, 1), (-1, -1), (2, 0)
    ]
    
    init_params = np.zeros(3*N)
    for idx, (i, j) in enumerate(lattice_coords):
        x, y = i * a1[0] + j * a2[0], i * a1[1] + j * a2[1]
        init_params[3*idx] = x
        init_params[3*idx+1] = y
        init_params[3*idx+2] = 0.0 # Angle 0

    # 2. Optimize using Nelder-Mead
    # We run multiple iterations/restarts if time permits, but Nelder-Mead is fast for N=12
    best_params = init_params.copy()
    best_val = objective_function(best_params)
    
    t_start = time.time()
    time_limit = DEADLINE - 5.0
    
    # Run Nelder-Mead
    try:
        res = minimize(objective_function, best_params, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass

    # If there is time, try a second run with slightly perturbed initial conditions
    # to escape local minima
    if time.time() - t_start < time_limit * 0.5:
        perturbed_params = best_params.copy()
        # Small random perturbation
        noise = np.random.normal(0, 0.1, 3*N)
        # Do not perturb angles too much initially, but positions can move
        for i in range(N):
            perturbed_params[3*i] += noise[3*i]
            perturbed_params[3*i+1] += noise[3*i+1]
            # Keep angles close to 0 or random? Let's keep them 0 for symmetry breaking
            # perturbed_params[3*i+2] = noise[3*i+2] * 5.0
        
        try:
            res2 = minimize(objective_function, perturbed_params, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res2.fun < best_val:
                best_params = res2.x
                best_val = res2.fun
        except Exception:
            pass

    # Extract results
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(N):
        pos[i, 0] = best_params[3*i]
        pos[i, 1] = best_params[3*i+1]
        ang[i] = best_params[3*i+2]
        
    # Final verification and centering
    # Check validity
    valid = True
    angles_rad = np.radians(ang)
    for i in range(N):
        for j in range(i+1, N):
            h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
            h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
            if not hexagons_disjoint_fast(h1, h2):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to initial honeycomb
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx, 0] = i * a1[0] + j * a2[0]
            pos[idx, 1] = i * a1[1] + j * a2[1]
            ang[idx] = 0.0

    side, phi, cx, cy = compute_min_outer_side(pos, ang)
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(ang[i])])
        
    center = (float(cx), float(cy))
    side = float(side)
    angle = float(phi)
    
    return inner, center, side, angle
