import math
import time
import numpy as np
from scipy.optimize import minimize, SLSQP

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap (interiors intersect or share boundary),
    positive if separated.
    """
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _min_enclosing_hex_side(inner_params, outer_angle_deg, smooth_beta=0.0):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon, given a fixed outer angle.
    
    For a regular hexagon centered at origin with side s and rotation angle_deg,
    a point (x,y) is inside iff for each of the 6 edge normals n_k,
    x*n_kx + y*n_ky <= apothem = s * sqrt(3)/2.
    
    So s >= max over all vertices v and all k of (v · n_k) / (sqrt(3)/2).
    """
    n = len(inner_params)
    apothem_factor = math.sqrt(3) / 2.0
    
    # Edge normals for outer hexagon with rotation outer_angle_deg
    normal_angles = np.deg2rad(outer_angle_deg + 30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Collect all vertex projections
    all_projections = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        projections = normals @ hex_v.T  # shape (6, 6)
        all_projections.append(projections)
    
    all_projs = np.concatenate(all_projections, axis=1)  # shape (6, 6*n)
    
    # Only consider positive projections
    max_val = 0.0
    for k in range(6):
        row = all_projs[k, :]
        pos_mask = row > 0
        if np.any(pos_mask):
            if smooth_beta > 0:
                # Smooth max using log-sum-exp
                pos_vals = row[pos_mask]
                log_sum = np.log(np.sum(np.exp(pos_vals / smooth_beta))) * smooth_beta
                required_s = log_sum / apothem_factor
            else:
                required_s = np.max(row[pos_mask]) / apothem_factor
            if required_s > max_val:
                max_val = required_s
    
    return max_val

def _compute_side(inner_params, outer_angle_deg, smooth_beta=0.0):
    """Compute minimum side length for given inner params and outer angle."""
    return _min_enclosing_hex_side(inner_params, outer_angle_deg, smooth_beta)

def _overlap_penalty(inner_params, penalty_weight=50.0, smooth_beta=0.0):
    """Compute penalty for overlapping inner hexagons."""
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < 0.005:
                if smooth_beta > 0:
                    # Smooth penalty
                    penalty += penalty_weight * (0.005 - sep) ** 2
                else:
                    penalty += penalty_weight * (0.005 - sep) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n, smooth_beta=0.0):
    """Objective: outer side length + overlap penalty."""
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg, smooth_beta)
    overlap_pen = _overlap_penalty(inner_params, smooth_beta=smooth_beta)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if configuration is valid."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
    
    return True

def _seed_solution():
    """Concentric ring seed solution."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def _lattice_packing():
    """
    Generate a hexagonal lattice packing of 11 unit hexagons.
    The 11 hexagons form a centered hexagonal cluster: 1 center + 6 ring + 4 more.
    """
    sqrt3 = math.sqrt(3)
    # Hexagonal lattice basis
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # Pick: center + 6 neighbors + (1,1),(-1,-1),(2,0),(0,-2)
    selected = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,0), (0,-2)]
    
    centers = []
    for (i, j) in selected:
        pos = i * v1 + j * v2
        centers.append(pos)
    
    # All hexagons aligned at 30 degrees (flat-top orientation)
    inner = []
    for pos in centers:
        inner.append([float(pos[0]), float(pos[1]), 30.0])
    
    # Compute bounding side
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in np.arange(0, 60, 1.0):
        side = _compute_side(np.array(inner), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    # Refine angle
    for angle_deg in np.arange(best_angle - 2, best_angle + 2, 0.1):
        side = _compute_side(np.array(inner), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return inner, (0.0, 0.0), best_side, best_angle

def _high_precision_polish(inner_params, outer_angle_deg, n, time_budget):
    """
    High-precision polish pass using SLSQP with tighter tolerances.
    Uses auxiliary variable t for epigraph formulation.
    """
    start_time = time.time()
    
    # Define objective with auxiliary variable t (side length)
    # x = [inner_params (3n), t]
    def epigraph_objective(x):
        if time.time() - start_time > time_budget:
            return float('inf')
        
        inner_p = x[:3*n].reshape(n, 3)
        t = x[3*n]
        
        # Constraint: side <= t
        actual_side = _compute_side(inner_p, outer_angle_deg)
        if actual_side > t + 1e-10:
            return float('inf')
        
        # Overlap penalty
        overlap_pen = _overlap_penalty(inner_p)
        
        return t + overlap_pen
    
    # Constraints: no overlap (sep >= 0)
    def no_overlap_constraint(x):
        if time.time() - start_time > time_budget:
            return 0.0
        
        inner_p = x[:3*n].reshape(n, 3)
        
        min_sep = float('inf')
        for i in range(n):
            v1 = _hex_vertices(inner_p[i,0], inner_p[i,1], inner_p[i,2])
            for j in range(i+1, n):
                v2 = _hex_vertices(inner_p[j,0], inner_p[j,1], inner_p[j,2])
                sep = _poly_separation(v1, v2)
                if sep < min_sep:
                    min_sep = sep
        
        return min_sep
    
    # Initial guess
    x0 = np.zeros(3*n + 1)
    x0[:3*n] = inner_params.flatten()
    x0[3*n] = _compute_side(inner_params, outer_angle_deg) + 0.1
    
    # Bounds
    bounds = [(-10, 10)] * 3*n + [(0, 20)]
    
    # Try SLSQP
    try:
        res = minimize(
            epigraph_objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': no_overlap_constraint}
            ],
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        
        if res.success and time.time() - start_time < time_budget:
            best_inner = res.x[:3*n].reshape(n, 3)
            best_side = _compute_side(best_inner, outer_angle_deg)
            
            if _is_valid(best_inner, best_side, outer_angle_deg, min_sep=1e-8):
                return best_inner, best_side
        
    except Exception:
        pass
    
    # Fallback: Nelder-Mead with very tight tolerances
    try:
        init_params = inner_params.flatten().copy()
        
        def nm_objective(params):
            if time.time() - start_time > time_budget:
                return float('inf')
            inner_p = params.reshape(n, 3)
            side = _compute_side(inner_p, outer_angle_deg)
            overlap_pen = _overlap_penalty(inner_p)
            return side + overlap_pen
        
        res_nm = minimize(
            nm_objective,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if time.time() - start_time < time_budget:
            best_inner = res_nm.x.reshape(n, 3)
            best_side = _compute_side(best_inner, outer_angle_deg)
            
            if _is_valid(best_inner, best_side, outer_angle_deg, min_sep=1e-8):
                return best_inner, best_side
        
    except Exception:
        pass
    
    return None, None

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    
    # Try multiple seed configurations
    seed_candidates = []
    
    # Seed 1: concentric ring
    seed1_inner, seed1_center, seed1_side, seed1_angle = _seed_solution()
    seed_candidates.append((seed1_inner, seed1_side, seed1_angle))
    
    # Seed 2: lattice packing
    seed2_inner, seed2_center, seed2_side, seed2_angle = _lattice_packing()
    seed_candidates.append((seed2_inner, seed2_side, seed2_angle))
    
    # Pick best seed
    best_seed = min(seed_candidates, key=lambda x: x[1])
    best_inner = [list(row) for row in best_seed[0]]
    best_side = best_seed[1]
    best_angle = best_seed[2]
    
    # Optimization: separate inner positions and outer angle
    # Phase 1: Optimize inner positions for fixed outer angle with smooth objective
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(best_seed[0]):
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = a
    
    # Try multiple outer angles
    outer_angles_to_try = [best_angle, 0.0, 15.0, 30.0, 45.0, 60.0]
    
    # Use smooth objective for first pass
    for outer_angle in outer_angles_to_try:
        if time.time() > DEADLINE - safety_margin:
            break
        
        try:
            # First pass with smooth beta
            res = minimize(_objective, init_params.copy(), args=(outer_angle, n, 0.1), 
                          method='Nelder-Mead', 
                          options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if time.time() > DEADLINE - safety_margin:
                break
            
            candidate_inner = res.x.reshape(n, 3)
            candidate_side = _compute_side(candidate_inner, outer_angle)
            
            # Check validity
            if _is_valid(candidate_inner, candidate_side, outer_angle, min_sep=1e-8):
                if candidate_side < best_side:
                    best_inner = candidate_inner.tolist()
                    best_side = candidate_side
                    best_angle = outer_angle
        except Exception:
            pass
    
    # Phase 2: Fine-tune outer angle with best inner positions
    if time.time() < DEADLINE - safety_margin:
        inner_arr = np.array(best_inner)
        best_angle_fine = best_angle
        for angle_deg in np.arange(best_angle - 3, best_angle + 3, 0.5):
            side = _compute_side(inner_arr, angle_deg)
            if side < best_side:
                best_side = side
                best_angle_fine = angle_deg
        # Refine
        for angle_deg in np.arange(best_angle_fine - 0.5, best_angle_fine + 0.5, 0.05):
            side = _compute_side(inner_arr, angle_deg)
            if side < best_side:
                best_side = side
                best_angle_fine = angle_deg
        best_angle = best_angle_fine
    
    # Phase 3: One more round of inner optimization with refined angle (non-smooth)
    if time.time() < DEADLINE - safety_margin:
        try:
            init_params2 = np.zeros(3 * n)
            for i, (x, y, a) in enumerate(best_inner):
                init_params2[i*3] = x
                init_params2[i*3+1] = y
                init_params2[i*3+2] = a
            
            res2 = minimize(_objective, init_params2, args=(best_angle, n, 0.0),
                           method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if time.time() < DEADLINE - safety_margin:
                candidate_inner2 = res2.x.reshape(n, 3)
                candidate_side2 = _compute_side(candidate_inner2, best_angle)
                if _is_valid(candidate_inner2, candidate_side2, best_angle, min_sep=1e-8):
                    if candidate_side2 < best_side:
                        best_inner = candidate_inner2.tolist()
                        best_side = candidate_side2
        except Exception:
            pass
    
    # Phase 4: High-precision polish pass
    if time.time() < DEADLINE - safety_margin:
        remaining_time = DEADLINE - safety_margin - time.time()
        if remaining_time > 10:
            polish_time_budget = min(remaining_time - 5, 30)  # Use up to 30s for polish
            polished_inner, polished_side = _high_precision_polish(
                np.array(best_inner), best_angle, n, polish_time_budget
            )
            if polished_inner is not None and polished_side < best_side:
                best_inner = polished_inner.tolist()
                best_side = polished_side
    
    # Final angle refinement
    if time.time() < DEADLINE - safety_margin:
        inner_arr = np.array(best_inner)
        for angle_deg in np.arange(best_angle - 1, best_angle + 1, 0.1):
            side = _compute_side(inner_arr, angle_deg)
            if side < best_side:
                best_side = side
                best_angle = angle_deg
        for angle_deg in np.arange(best_angle - 0.1, best_angle + 0.1, 0.01):
            side = _compute_side(inner_arr, angle_deg)
            if side < best_side:
                best_side = side
                best_angle = angle_deg
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
    
    center = (0.0, 0.0)
    side = float(best_side)
    angle_deg = float(best_angle)
    
    return inner_rows, center, side, angle_deg
