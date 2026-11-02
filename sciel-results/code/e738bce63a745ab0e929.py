import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill

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
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
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
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        en = edge_normals
        en1x = en[:, 0] * c1 - en[:, 1] * s1
        en1y = en[:, 0] * s1 + en[:, 1] * c1
        en2x = en[:, 0] * c2 - en[:, 1] * s2
        en2y = en[:, 0] * s2 + en[:, 1] * c2
        
        eps = 1e-9
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() - eps < p2.min() or p2.max() - eps < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() - eps < p2.min() or p2.max() - eps < p1.min():
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
                                  options={'xatol': 1e-10})
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

    def compute_continuous_overlap_penalty(pos, angles_rad, sharpness=100.0):
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 3.0:
                    overlap = max(0.0, 2.0 - dist)
                    penalty += math.exp(sharpness * overlap) / sharpness
        return penalty

    def symmetric_objective(params):
        # params: [x1, y1, a1_deg, x2, y2, a2_deg]
        x1, y1, a1_deg = params[0], params[1], params[2]
        x2, y2, a2_deg = params[3], params[4], params[5]
        
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        
        for k in range(6):
            theta = np.radians(60 * k)
            c, s = math.cos(theta), math.sin(theta)
            
            # Orbit 1
            px1 = x1 * c - y1 * s
            py1 = x1 * s + y1 * c
            pa1 = a1_deg + 60 * k
            positions[2*k] = [px1, py1]
            angles_deg[2*k] = pa1
            
            # Orbit 2
            px2 = x2 * c - y2 * s
            py2 = x2 * s + y2 * c
            pa2 = a2_deg + 60 * k
            positions[2*k+1] = [px2, py2]
            angles_deg[2*k+1] = pa2
            
        angles_rad = np.radians(angles_deg)
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    # Heavy penalty
                    return 100.0 + compute_continuous_overlap_penalty(positions, angles_rad, sharpness=50.0)
        
        side, phi, cx, cy = compute_min_outer_side(positions, angles_deg)
        return side

    def generate_symmetric_initial_configs():
        configs = []
        # Try some heuristic starting points
        # 1. Lattice-like
        configs.append([1.5, 0.8, 0, 1.5, -0.8, 0])
        configs.append([1.5, 0.8, 30, 1.5, -0.8, -30])
        configs.append([2.0, 1.0, 0, 2.0, -1.0, 0])
        configs.append([1.0, 1.0, 0, 2.0, 0.0, 0])
        
        # Random
        for _ in range(10):
            configs.append([
                random.uniform(0.5, 3.0), random.uniform(-2.0, 2.0), random.uniform(0, 60),
                random.uniform(0.5, 3.0), random.uniform(-2.0, 2.0), random.uniform(0, 60)
            ])
        return configs

    best_side = float('inf')
    best_params = None
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Phase 1: Differential Evolution on symmetric parameters
    bounds = [
        (-3.0, 3.0), (-3.0, 3.0), (0.0, 60.0),
        (-3.0, 3.0), (-3.0, 3.0), (0.0, 60.0)
    ]
    
    de_start = time.time()
    try:
        res = differential_evolution(
            symmetric_objective,
            bounds,
            maxiter=200,
            popsize=30,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
            init='latinhypercube'
        )
        
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
    except Exception:
        pass

    # Phase 2: Local refinement with Nelder-Mead from best DE result and random starts
    if time.time() - t0 < DEADLINE - 10:
        starts = []
        if best_params is not None:
            starts.append(best_params)
        starts.extend(generate_symmetric_initial_configs()[:5])
        
        for start in starts:
            if time.time() - t0 > DEADLINE - 5:
                break
            try:
                res = minimize(symmetric_objective, start, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_side:
                    best_side = res.fun
                    best_params = res.x
            except Exception:
                pass

    # Reconstruct best configuration
    if best_params is not None:
        x1, y1, a1_deg = best_params[0], best_params[1], best_params[2]
        x2, y2, a2_deg = best_params[3], best_params[4], best_params[5]
        
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        
        for k in range(6):
            theta = np.radians(60 * k)
            c, s = math.cos(theta), math.sin(theta)
            
            px1 = x1 * c - y1 * s
            py1 = x1 * s + y1 * c
            pa1 = a1_deg + 60 * k
            positions[2*k] = [px1, py1]
            angles_deg[2*k] = pa1
            
            px2 = x2 * c - y2 * s
            py2 = x2 * s + y2 * c
            pa2 = a2_deg + 60 * k
            positions[2*k+1] = [px2, py2]
            angles_deg[2*k+1] = pa2
            
        if is_valid_configuration(positions, angles_deg):
            best_positions = positions
            best_angles = angles_deg
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(positions, angles_deg)
        else:
            # Fallback to a known valid config if symmetry search failed
            print("Symmetric config invalid, falling back")
            a1 = np.array([1.5 * 1.001, (sqrt3/2.0) * 1.001])
            a2 = np.array([0.0 * 1.001, sqrt3 * 1.001])
            lattice_coords = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, -1), (-1, 1), (2, -1), (-2, 1),
                (1, 1), (-1, -1), (2, 0),
            ]
            best_positions = np.zeros((N, 2))
            best_angles = np.zeros(N)
            for idx, (i, j) in enumerate(lattice_coords):
                best_positions[idx] = i * a1 + j * a2
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    else:
        # Fallback
        a1 = np.array([1.5 * 1.001, (sqrt3/2.0) * 1.001])
        a2 = np.array([0.0 * 1.001, sqrt3 * 1.001])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            best_positions[idx] = i * a1 + j * a2
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
