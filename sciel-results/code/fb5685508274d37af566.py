import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3 + np.pi / 6
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _sat_overlap(va, vb, normals_a, normals_b):
    """Check if two hexagons (given as vertex arrays) overlap using SAT.
    Returns the minimum separation (negative if overlapping)."""
    min_sep = float('inf')
    for n in np.vstack([normals_a, normals_b]):
        proj_a = va @ n
        proj_b = vb @ n
        gap = max(np.min(proj_a) - np.max(proj_b), np.min(proj_b) - np.max(proj_a))
        if gap < min_sep:
            min_sep = gap
    return min_sep

def _compute_s_and_overlap_fast(positions, angle_deg):
    """Compute minimum outer hexagon side length and check for overlaps.
    Returns (s, min_overlap) where min_overlap < 0 means overlap exists."""
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    unit_normals = _edge_normals(0, 0, 1, 0)
    
    # Compute all vertices
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    # Compute normals for outer hexagon
    a_rad = np.deg2rad(angle_deg)
    outer_normals = np.column_stack([
        np.cos(a_rad + i * np.pi / 3 + np.pi / 6),
        np.sin(a_rad + i * np.pi / 3 + np.pi / 6)
    ] for i in range(6))
    
    # Compute s
    projections = all_verts_flat @ outer_normals.T
    max_projections = np.max(projections, axis=0)
    s = np.max(max_projections) * 2 / np.sqrt(3)
    
    # Check overlaps
    min_overlap = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            vi = all_verts[i]
            vj = all_verts[j]
            sep = _sat_overlap(vi, vj, unit_normals, unit_normals)
            if sep < min_overlap:
                min_overlap = sep
    
    return s, min_overlap

def _resolve_collisions(positions, angle_deg, max_iter=200, tol=1e-10):
    """Resolve collisions by iteratively pushing apart overlapping hexagons."""
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    unit_normals = _edge_normals(0, 0, 1, 0)
    positions = positions.copy()
    
    for _ in range(max_iter):
        # Find maximum overlap
        max_overlap = 0.0
        max_pair = None
        max_direction = None
        
        for i in range(12):
            for j in range(i + 1, 12):
                vi = positions[i] + unit_hex_verts
                vj = positions[j] + unit_hex_verts
                sep = _sat_overlap(vi, vj, unit_normals, unit_normals)
                if sep < max_overlap:
                    max_overlap = sep
                    max_pair = (i, j)
                    # Find the normal that gives the maximum overlap
                    for n in np.vstack([unit_normals, unit_normals]):
                        proj_i = vi @ n
                        proj_j = vj @ n
                        gap = max(np.min(proj_i) - np.max(proj_j), np.min(proj_j) - np.max(proj_i))
                        if gap == sep:
                            max_direction = n
                            break
        
        if max_overlap >= -tol:
            break
        
        # Push apart along the direction of maximum overlap
        i, j = max_pair
        # Determine which way to push
        vi = positions[i] + unit_hex_verts
        vj = positions[j] + unit_hex_verts
        proj_i = vi @ max_direction
        proj_j = vj @ max_direction
        
        if np.mean(proj_i) < np.mean(proj_j):
            # i is on the negative side, push i further negative, j further positive
            positions[i] -= max_direction * (abs(max_overlap) / 2 + 1e-8)
            positions[j] += max_direction * (abs(max_overlap) / 2 + 1e-8)
        else:
            positions[i] += max_direction * (abs(max_overlap) / 2 + 1e-8)
            positions[j] -= max_direction * (abs(max_overlap) / 2 + 1e-8)
    
    return positions

def _generate_seeds():
    """Generate diverse initial seeds for the optimization."""
    seeds = []
    
    # Standard hexagonal packing
    h = np.sqrt(3) / 2
    base_pos = np.array([
        [0, 0],
        [2*h, 0],
        [h, 1.5],
        [-h, 1.5],
        [-2*h, 0],
        [-h, -1.5],
        [h, -1.5],
        [3*h, 1.5],
        [0, 3],
        [-3*h, 1.5],
        [-3*h, -1.5],
        [3*h, -1.5]
    ])
    seeds.append((base_pos, 0.0))
    
    # Perturbed versions
    np.random.seed(42)
    for _ in range(20):
        noise = np.random.randn(12, 2) * 0.3
        angle = np.random.uniform(-30, 30)
        seeds.append((base_pos + noise, angle))
    
    # Different lattice configurations
    for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
        pos = base_pos * scale
        angle = np.random.uniform(-15, 15)
        seeds.append((pos, angle))
    
    return seeds

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
            na = _edge_normals(inner[i][0], inner[i][1], 1, inner[i][2])
            nb = _edge_normals(inner[j][0], inner[j][1], 1, inner[j][2])
            sep = _sat_overlap(va, vb, na, nb)
            if sep < -1e-6:
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
    deadline = start_time + 570
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    def update_best(positions, angle_deg):
        nonlocal best_score, best_inner, best_outer, best_s
        s, min_overlap = _compute_s_and_overlap_fast(positions, angle_deg)
        
        if min_overlap < -1e-6:
            return
        
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = positions
        inner_candidate[:, 2] = 0.0
        outer_candidate = np.array([0, 0, angle_deg])
        
        if _check_validity(inner_candidate, outer_candidate, s):
            score_candidate = 1.0 / s
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate.copy()
                best_outer = outer_candidate.copy()
                best_s = s
    
    def objective(x):
        positions = x[:24].reshape(12, 2)
        angle_deg = x[24]
        s, min_overlap = _compute_s_and_overlap_fast(positions, angle_deg)
        penalty = 0.0
        if min_overlap < 0:
            penalty = (-min_overlap) ** 2 * 1000
        return s + penalty
    
    # Generate seeds
    seeds = _generate_seeds()
    
    # Phase 1: Collision resolution + optimization for each seed
    for seed_pos, seed_angle in seeds:
        if time.time() > deadline:
            break
        
        # Resolve collisions
        resolved_pos = _resolve_collisions(seed_pos, seed_angle, max_iter=100)
        
        # Update with resolved position
        update_best(resolved_pos, seed_angle)
        
        # Optimize from resolved position
        x0 = np.concatenate([resolved_pos.flatten(), [seed_angle]])
        
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            positions = res.x[:24].reshape(12, 2)
            angle_deg = res.x[24]
            update_best(positions, angle_deg)
        except Exception:
            continue
    
    # Phase 2: Fine-tuning with multiple outer angle starts
    if best_inner is not None:
        best_positions = best_inner[:, :2].copy()
        best_angle = best_outer[2]
        
        for angle_start in [best_angle - 10, best_angle - 5, best_angle, best_angle + 5, best_angle + 10]:
            if time.time() > deadline:
                break
            
            x0 = np.concatenate([best_positions.flatten(), [angle_start]])
            
            try:
                res = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                
                positions = res.x[:24].reshape(12, 2)
                angle_deg = res.x[24]
                update_best(positions, angle_deg)
            except Exception:
                continue
    
    # Phase 3: Final high-precision polish
    if best_inner is not None and time.time() < deadline - 30:
        best_positions = best_inner[:, :2].copy()
        best_angle = best_outer[2]
        
        x0 = np.concatenate([best_positions.flatten(), [best_angle]])
        
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            positions = res.x[:24].reshape(12, 2)
            angle_deg = res.x[24]
            update_best(positions, angle_deg)
        except Exception:
            pass
    
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback: standard hexagonal packing
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
        s_fallback, _ = _compute_s_and_overlap_fast(inner_fallback[:, :2], 0)
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
