import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    projections = all_verts_flat @ normals.T
    max_projections = np.max(projections, axis=0)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    return min_s

def _check_validity(inner, outer, s):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            separated = False
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    # 12 lattice coordinates in a hexagonal arrangement
    # (m, n) pairs for triangular lattice: position = m*a1 + n*a2
    lattice_coords = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [-1, 1],
        [1, -1],
        [2, 0],
        [-2, 0],
        [2, -1],
        [-2, 1],
        [1, 1]
    ], dtype=float)
    
    def _lattice_positions(x):
        # x = [a1x, a1y, a2x, a2y, theta]
        a1 = np.array([x[0], x[1]])
        a2 = np.array([x[2], x[3]])
        theta = x[4]
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        a1_rot = R @ a1
        a2_rot = R @ a2
        positions = lattice_coords[:, 0] * a1_rot + lattice_coords[:, 1] * a2_rot
        return positions
    
    def _overlap_penalty(positions):
        # Smooth penalty for overlapping hexagons
        # For unit hexagons on a lattice, minimum center distance for no overlap
        # depends on relative orientation. Use a conservative threshold.
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                # For aligned hexagons, edge-to-edge touching is at d=1.5
                # Use a smooth penalty below 1.5
                if d < 1.5:
                    penalty += (1.5 - d)**2
                elif d < 1.6:
                    penalty += 0.05 * (1.6 - d)**2
        return penalty
    
    def objective(x):
        positions = _lattice_positions(x)
        # Optimize outer angle to minimize s
        def outer_obj(theta):
            s = _min_outer_side(positions, theta)
            return s
        res = minimize(outer_obj, np.array([0.0]), method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        s = res.fun
        penalty = _overlap_penalty(positions)
        return s + penalty * 50.0
    
    # Initial guess: regular triangular lattice
    h = np.sqrt(3)/2
    x0 = np.array([2.0, 0.0, 1.0, h, 0.0])
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Multi-start optimization
    np.random.seed(123)
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            x_start = x0.copy()
        else:
            # Perturb around the regular lattice
            noise = np.random.randn(5) * [0.3, 0.3, 0.3, 0.3, 0.1]
            x_start = x0 + noise
        
        try:
            res = minimize(objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        except Exception:
            continue
        
        x_opt = res.x
        positions = _lattice_positions(x_opt)
        
        # Find optimal outer angle
        def outer_obj(theta):
            return _min_outer_side(positions, theta)
        res_outer = minimize(outer_obj, np.array([0.0]), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
        angle_opt = res_outer.x[0]
        s_candidate = res_outer.fun
        
        # Build inner array (all hexagons aligned at 0)
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = positions
        inner_candidate[:, 2] = 0.0
        
        # Center the configuration
        center = np.mean(positions, axis=0)
        inner_candidate[:, :2] -= center
        outer_candidate = np.array([0.0, 0.0, angle_opt])
        
        # Recompute s after centering
        s_candidate = _min_outer_side(inner_candidate[:, :2], angle_opt)
        
        if _check_validity(inner_candidate, outer_candidate, s_candidate):
            score_candidate = 1.0 / s_candidate
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate.copy()
                best_outer = outer_candidate.copy()
                best_s = s_candidate
    
    # If we found a valid solution, return it
    if best_inner is not None:
        return best_inner, best_outer, best_s
    
    # Fallback: regular hexagonal packing
    h = np.sqrt(3)/2
    inner_fallback = np.array([
        [0, 0, 0],
        [2*h, 0, 0],
        [h, 1.5, 0],
        [-h, 1.5, 0],
        [-2*h, 0, 0],
        [-h, -1.5, 0],
        [h, -1.5, 0],
        [3*h, 1.5, 0],
        [0, 3, 0],
        [-3*h, 1.5, 0],
        [-3*h, -1.5, 0],
        [3*h, -1.5, 0]
    ])
    outer_fallback = np.array([0, 0, 0])
    s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
    if s_fallback < np.sqrt(12):
        s_fallback = np.sqrt(12)
    return inner_fallback, outer_fallback, s_fallback
