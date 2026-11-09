# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))

def _hex_edge_normals(vertices):
    """Return outward normals of the 6 edges of a hexagon given its 6 vertices."""
    n = 6
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Outward normal: rotate edge by -90 degrees (clockwise) and normalize
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normals[i] = normal / norm
    return normals

def _min_separation(v1, v2):
    """Compute the minimum separation between two hexagons. Positive if separated, negative if overlapping."""
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_sep = np.inf
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            sep = -min(max1 - min2, max2 - min1)
        
        min_sep = min(min_sep, sep)
        if min_sep < -1e-6:  # Early exit if definitely overlapping
            return min_sep
    
    return min_sep

def _compute_s(inner_positions, outer_center, outer_angle_deg, inner_angles=None):
    """Compute the minimum side length s of outer hexagon containing all inner hexagons."""
    outer_center = np.array(outer_center)
    inner_centers = np.array(inner_positions)
    if inner_angles is None:
        inner_angles = np.zeros(len(inner_centers))
    
    shifted = inner_centers - outer_center
    
    a = np.radians(outer_angle_deg)
    normal_angles = a + np.radians(60) * np.arange(6) + np.radians(30)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    all_vertices = []
    for i in range(len(inner_centers)):
        cx, cy = shifted[i]
        ang = inner_angles[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices) # (72, 2)
    
    # Project all vertices onto all outer normals
    projections = all_vertices @ normals.T
    
    max_proj = projections.max()
    
    s = 2 * max_proj / np.sqrt(3)
    
    return s

def _is_inside_outer(inner_vertices, outer_vertices):
    """Check if all vertices of inner hex are inside outer hex."""
    n_out = len(outer_vertices)
    for v in inner_vertices:
        inside = True
        for i in range(n_out):
            p1 = outer_vertices[i]
            p2 = outer_vertices[(i + 1) % n_out]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                inside = False
                break
        if not inside:
            return False
    return True

def _check_validity(pos, outer_center, outer_angle, inner_angles, s):
    """Check if the packing is valid. Returns True if valid."""
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(pos[i,0], pos[i,1], 1, inner_angles[i])
            v2 = _hex_vertices(pos[j,0], pos[j,1], 1, inner_angles[j])
            min_sep = _min_separation(v1, v2)
            if min_sep < -1e-6:
                return False
    
    # Check inside outer
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
    for i in range(12):
        inner_verts = _hex_vertices(pos[i,0], pos[i,1], 1, inner_angles[i])
        if not _is_inside_outer(inner_verts, outer_verts):
            return False
    
    return True

def _fit_outer_hex(pos, inner_angles=None):
    """Fit the outer hexagon to minimize s for given inner positions."""
    if inner_angles is None:
        inner_angles = np.zeros(12)
    
    def obj_outer(params_outer):
        oc = params_outer[:2]
        oa = params_outer[2]
        return _compute_s(pos, oc, oa, inner_angles)
    
    # Try multiple initial angles
    best_s = np.inf
    best_params = None
    
    for oa_init in [0, 30, 60, 90]:
        oc_init = np.mean(pos, axis=0)
        res_outer = minimize(obj_outer, np.concatenate([oc_init, [oa_init]]), method='Nelder-Mead', 
                             options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 300})
        if res_outer.fun < best_s:
            best_s = res_outer.fun
            best_params = res_outer.x
    
    return best_s, best_params

def _overlap_penalty(pos, inner_angles):
    """Compute total overlap penalty."""
    penalty = 0
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(pos[i,0], pos[i,1], 1, inner_angles[i])
            v2 = _hex_vertices(pos[j,0], pos[j,1], 1, inner_angles[j])
            min_sep = _min_separation(v1, v2)
            if min_sep < 0:
                penalty += 100 * min_sep * min_sep
    return penalty

def _objective_positions(params_pos, inner_angles):
    """Objective function for position optimization."""
    pos = params_pos.reshape(12, 2)
    
    penalty = _overlap_penalty(pos, inner_angles)
    if penalty > 10:
        return 1000 + penalty
    
    s, _ = _fit_outer_hex(pos, inner_angles)
    return s + penalty

def _make_seed_lattice(rotation=0):
    """Create a honeycomb lattice seed."""
    # Distance between centers of touching hexagons is sqrt(3)
    d = np.sqrt(3)
    
    # Hexagonal arrangement
    positions = []
    # Center
    positions.append([0, 0])
    
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = np.radians(60 * i + rotation)
        positions.append([d * np.cos(angle), d * np.sin(angle)])
    
    # Ring 2: 6 more hexagons (to make 12 total)
    for i in range(6):
        angle = np.radians(60 * i + 30 + rotation)
        r = d * np.sqrt(3)
        positions.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(positions)

def _make_seed_cluster():
    """Create a compact cluster seed."""
    d = np.sqrt(3)
    positions = [
        [0, 0],
        [-d, 0],
        [d, 0],
        [-d/2, d*np.sqrt(3)/2],
        [d/2, d*np.sqrt(3)/2],
        [-d/2, -d*np.sqrt(3)/2],
        [d/2, -d*np.sqrt(3)/2],
        [-3*d/2, d*np.sqrt(3)/2],
        [3*d/2, d*np.sqrt(3)/2],
        [-3*d/2, -d*np.sqrt(3)/2],
        [3*d/2, -d*np.sqrt(3)/2],
        [0, -d*np.sqrt(3)]
    ]
    return np.array(positions)

def _make_seed_two_row():
    """Create a two-row seed."""
    d = np.sqrt(3)
    positions = []
    # Row 1: 6 hexagons
    for i in range(6):
        positions.append([(i - 2.5) * d, 0])
    # Row 2: 6 hexagons offset
    for i in range(6):
        positions.append([(i - 2.5) * d + d/2, d*np.sqrt(3)/2])
    return np.array(positions)

def _make_seed_triangular():
    """Create a triangular arrangement seed."""
    d = np.sqrt(3)
    positions = []
    # Triangle with side 3
    for i in range(3):
        for j in range(3 - i):
            x = j * d + i * d/2
            y = -i * d * np.sqrt(3)/2
            positions.append([x, y])
    # Add 3 more to make 12
    positions.append([0, d*np.sqrt(3)])
    positions.append([d, d*np.sqrt(3)])
    positions.append([d/2, d*np.sqrt(3) + d/2])
    return np.array(positions)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600
    
    # Multiple diverse seeds
    seeds = [
        _make_seed_cluster(),
        _make_seed_lattice(0),
        _make_seed_lattice(30),
        _make_seed_two_row(),
        _make_seed_triangular()
    ]
    
    best_params = None
    best_score = np.inf
    best_valid = False
    
    # Initialize with the best seed
    for seed in seeds:
        s, outer_params = _fit_outer_hex(seed, np.zeros(12))
        if s < best_score:
            best_score = s
            best_params = np.concatenate([seed.flatten(), outer_params])
    
    # Phase 1: Simulated Annealing
    def _sa_step(params, temperature):
        pos = params[:24].reshape(12, 2)
        outer_center = params[24:26]
        outer_angle = params[26]
        
        # Perturb positions
        new_pos = pos + np.random.normal(0, 0.1, pos.shape)
        new_params = np.concatenate([new_pos.flatten(), outer_center, [outer_angle]])
        
        old_s = _compute_s(pos, outer_center, outer_angle, np.zeros(12))
        old_penalty = _overlap_penalty(pos, np.zeros(12))
        old_score = old_s + old_penalty
        
        new_s = _compute_s(new_pos, outer_center, outer_angle, np.zeros(12))
        new_penalty = _overlap_penalty(new_pos, np.zeros(12))
        new_score = new_s + new_penalty
        
        delta = new_score - old_score
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            return new_params
        return params
    
    # Run SA for each seed
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 120:
            break
        
        params = np.concatenate([seed.flatten(), np.array([0, 0]), [0]])
        
        # Fit outer hex first
        s, outer_params = _fit_outer_hex(seed, np.zeros(12))
        params = np.concatenate([seed.flatten(), outer_params])
        
        # SA iterations
        temperature = 0.5
        for it in range(200):
            if time.time() > deadline - 120:
                break
            params = _sa_step(params, temperature)
            temperature *= 0.98
        
        # Evaluate
        pos = params[:24].reshape(12, 2)
        s, outer_params = _fit_outer_hex(pos, np.zeros(12))
        score = s
        if score < best_score:
            best_score = score
            best_params = np.concatenate([pos.flatten(), outer_params])
    
    # Phase 2: Nelder-Mead optimization
    for restart in range(5):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            init_pos = best_params[:24].reshape(12, 2)
        else:
            init_pos = best_params[:24].reshape(12, 2) + np.random.normal(0, 0.3, (12, 2))
        
        try:
            res = minimize(_objective_positions, init_pos.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 300, 'maxfev': 1500})
            
            if res.fun < best_score:
                best_pos = res.x.reshape(12, 2)
                s_best, outer_params_best = _fit_outer_hex(best_pos, np.zeros(12))
                if s_best < best_score:
                    best_score = s_best
                    best_params = np.concatenate([best_pos.flatten(), outer_params_best])
                    
        except Exception:
            pass
    
    # Phase 3: Fine-tuning
    if time.time() < deadline - 30:
        pos = best_params[:24].reshape(12, 2)
        
        # Try to optimize each position individually
        for i in range(12):
            if time.time() > deadline - 30:
                break
            
            def obj_single(params_single):
                new_pos = pos.copy()
                new_pos[i] = params_single
                penalty = _overlap_penalty(new_pos, np.zeros(12))
                if penalty > 10:
                    return 1000 + penalty
                s, _ = _fit_outer_hex(new_pos, np.zeros(12))
                return s + penalty
            
            try:
                res = minimize(obj_single, pos[i], method='Nelder-Mead',
                               options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100})
                if res.fun < best_score:
                    pos[i] = res.x
                    s_check, _ = _fit_outer_hex(pos, np.zeros(12))
                    if s_check < best_score:
                        best_score = s_check
                        best_params = np.concatenate([pos.flatten(), np.zeros(2), [0]])
            except Exception:
                pass
        
        # Refit outer hex
        pos = best_params[:24].reshape(12, 2)
        s_best, outer_params_best = _fit_outer_hex(pos, np.zeros(12))
        if s_best < best_score:
            best_score = s_best
            best_params = np.concatenate([pos.flatten(), outer_params_best])
    
    # Extract final solution
    best_pos = best_params[:24].reshape(12, 2)
    best_outer_center = best_params[24:26]
    best_outer_angle = best_params[26]
    best_angles = np.zeros(12)
    
    # Calculate final s
    final_s = _compute_s(best_pos, best_outer_center, best_outer_angle, best_angles)
    
    # Check validity
    valid = _check_validity(best_pos, best_outer_center, best_outer_angle, best_angles, final_s)
    
    if not valid:
        # Fall back to best valid seed
        for seed in seeds:
            s, outer_params = _fit_outer_hex(seed, np.zeros(12))
            pos = seed
            oc = outer_params[:2]
            oa = outer_params[2]
            if _check_validity(pos, oc, oa, np.zeros(12), s):
                best_pos = pos
                best_outer_center = oc
                best_outer_angle = oa
                best_angles = np.zeros(12)
                final_s = s
                break
    
    # Construct return values
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_pos[i, 0]
        inner_hex_data[i, 1] = best_pos[i, 1]
        inner_hex_data[i, 2] = best_angles[i]
    
    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    return inner_hex_data, outer_hex_data, final_s

# EVOLVE-BLOCK-END
