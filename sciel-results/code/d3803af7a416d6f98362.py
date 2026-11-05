import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 145.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _all_vertices(inner_params):
    """Compute all vertices for all inner hexagons. Returns (n, 6, 2)."""
    n = len(inner_params)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
    return verts

def _compute_side_fast(verts, outer_angle_deg):
    """
    Fast computation of minimum outer hexagon side length.
    verts: (n, 6, 2) array of all inner hexagon vertices.
    outer_angle_deg: rotation of outer hexagon.
    Returns the minimum side length S.
    """
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    # Face normals of outer hexagon (perpendicular to edges)
    normal_angles = outer_angle_rad + np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # Flatten vertices: (n*6, 2)
    flat = verts.reshape(-1, 2)
    
    # For each normal, compute max dot product over all points
    # dots: (6, n*6)
    dots = normals @ flat.T
    max_dots = np.max(dots, axis=1)  # (6,)
    
    # side = max(max_dots) / apothem, where apothem = sqrt(3)/2
    return np.max(max_dots) / (SQRT3 / 2.0)

def _compute_side(inner_params, outer_angle_deg):
    """Compute minimum outer side length from inner params."""
    verts = _all_vertices(inner_params)
    return _compute_side_fast(verts, outer_angle_deg)

def _soft_max(vals, beta=80.0):
    """Smooth max via log-sum-exp."""
    m = np.max(vals)
    if m < -1e10:
        return m
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta

def _support_function_smooth(inner_params, outer_angle_deg):
    """Smooth support function for gradient-based optimization."""
    n = len(inner_params)
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle_rad + np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    all_pts = []
    for i in range(n):
        x, y, a = inner_params[i]
        vs = _hex_vertices(x, y, a)
        all_pts.append(vs)
    all_pts = np.vstack(all_pts)
    
    dots = normals @ all_pts.T  # (6, 6n)
    max_dots = np.max(dots, axis=1)  # (6,)
    sm = _soft_max(max_dots, beta=80.0)
    return sm / (SQRT3 / 2.0)

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
    """Minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = f / e
        t = max(0, min(1, t))
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = c / a
        s = max(0, min(1, s))
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
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
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    else:
        t = f / e
        t = max(0, min(1, t))
        return np.linalg.norm(p1 - (p3 + t * d2))

def _poly_separation(p1, p2):
    """Minimum distance between two convex polygons. Returns 0 if overlapping."""
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

def _overlap_penalty(inner_params, penalty_weight=50.0):
    """Compute penalty for overlapping inner hexagons."""
    n = len(inner_params)
    verts = _all_vertices(inner_params)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            # Quick center distance check
            c1 = verts[i].mean(axis=0)
            c2 = verts[j].mean(axis=0)
            cd = np.linalg.norm(c1 - c2)
            if cd < 3.5:
                sep = _poly_separation(verts[i], verts[j])
                if sep < 0.01:
                    penalty += penalty_weight * (0.01 - sep) ** 2
    return penalty

def _objective(params, outer_angle_deg, n):
    """Objective: outer side length + overlap penalty."""
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _objective_smooth(params, outer_angle_deg, n):
    """Smooth objective for gradient-based methods."""
    inner_params = params.reshape(n, 3)
    side = _support_function_smooth(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=1e-9):
    """Check if configuration is valid: all inside outer, no overlaps."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    verts = _all_vertices(inner_params)
    
    for i in range(n):
        for v in verts[i]:
            if not _point_in_polygon(v, outer_v):
                return False
    
    for i in range(n):
        for j in range(i+1, n):
            sep = _poly_separation(verts[i], verts[j])
            if sep < min_sep:
                return False
    return True

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _seed_solution():
    """Return the concentric ring seed solution."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (SQRT3 + eps) * np.cos(angle)
        y = (SQRT3 + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    max_x = (2 + eps) * SQRT3
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / SQRT3
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def _find_best_outer_angle(verts, n_coarse=20):
    """Find the best outer hexagon angle for given inner vertices."""
    def S_of_angle(oangle_deg):
        return _compute_side_fast(verts, oangle_deg)
    
    best_S = float('inf')
    best_angle = 0.0
    for i in range(n_coarse):
        ang = i * 180.0 / n_coarse  # 0 to 180 degrees
        s_val = S_of_angle(ang)
        if s_val < best_S:
            best_S = s_val
            best_angle = ang
    
    # Fine search
    try:
        result = minimize_scalar(S_of_angle, bounds=(best_angle - 5, best_angle + 5),
                                 method='bounded', options={'xatol': 1e-6})
        if result.fun < best_S:
            best_S = result.fun
            best_angle = result.x
    except Exception:
        pass
    
    return best_S, best_angle

def _morphing_search(inner_params, outer_angle_deg, n, time_limit=40.0):
    """
    Geometric morphing: parameterize as displacement from hexagonal lattice,
    optimize with smooth L-BFGS-B then exact Nelder-Mead.
    """
    start_time = time.time()
    
    # Hexagonal lattice positions for 11 hexagons
    lattice_positions = np.array([
        [0.0, 0.0, 30.0],
        [SQRT3, 0.0, 30.0], [-SQRT3, 0.0, 30.0],
        [SQRT3/2, 1.5, 30.0], [-SQRT3/2, 1.5, 30.0],
        [SQRT3/2, -1.5, 30.0], [-SQRT3/2, -1.5, 30.0],
        [SQRT3, 3.0, 30.0], [-SQRT3, 3.0, 30.0],
        [SQRT3, -3.0, 30.0], [-SQRT3, -3.0, 30.0],
    ])
    
    best_result = inner_params
    best_val = _compute_side(inner_params, outer_angle_deg)
    
    inits = [lattice_positions.copy(), inner_params.copy()]
    
    for init in inits:
        if time.time() - start_time > time_limit:
            break
        
        # Phase 1: Smooth L-BFGS-B
        try:
            res = minimize(_objective_smooth, init.flatten(), args=(outer_angle_deg, n),
                          method='L-BFGS-B',
                          options={'maxiter': 400, 'gtol': 1e-9, 'ftol': 1e-11})
            
            if time.time() - start_time > time_limit:
                break
            
            candidate = res.x.reshape(n, 3)
            cand_side = _compute_side(candidate, outer_angle_deg)
            if cand_side < best_val and _is_valid(candidate, cand_side, outer_angle_deg, min_sep=1e-9):
                best_result = candidate
                best_val = cand_side
        except Exception:
            pass
        
        # Phase 2: Exact Nelder-Mead
        if time.time() - start_time > time_limit - 5:
            break
        try:
            res2 = minimize(_objective, best_result.flatten(), args=(outer_angle_deg, n),
                           method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if time.time() - start_time > time_limit:
                break
            
            candidate = res2.x.reshape(n, 3)
            cand_side = _compute_side(candidate, outer_angle_deg)
            if cand_side < best_val and _is_valid(candidate, cand_side, outer_angle_deg, min_sep=1e-9):
                best_result = candidate
                best_val = cand_side
        except Exception:
            pass
    
    return best_result, best_val

def _psp_swarm(inner_params, outer_angle_deg, n, iterations=150, swarm_size=15, time_limit=25.0):
    """Particle Swarm Optimization."""
    start_time = time.time()
    best_global = inner_params.copy()
    best_global_val = _objective(best_global.flatten(), outer_angle_deg, n)
    
    particles = np.zeros((swarm_size, n, 3))
    velocities = np.zeros((swarm_size, n, 3))
    personal_bests = np.zeros((swarm_size, n, 3))
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        if i == 0:
            particles[i] = inner_params
        else:
            particles[i] = inner_params + np.random.randn(n, 3) * 0.5
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)
        
        val = _objective(particles[i].flatten(), outer_angle_deg, n)
        personal_bests[i] = particles[i]
        personal_best_vals[i] = val
        
        if val < best_global_val:
            best_global_val = val
            best_global = particles[i].copy()
    
    w = 0.85
    c1 = 1.2
    c2 = 1.2
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        for i in range(swarm_size):
            r1 = np.random.rand(n, 3)
            r2 = np.random.rand(n, 3)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_bests[i] - particles[i]) + c2 * r2 * (best_global - particles[i])
            velocities[i] = np.clip(velocities[i], -0.3, 0.3)
            
            particles[i] = particles[i] + velocities[i]
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)
            
            val = _objective(particles[i].flatten(), outer_angle_deg, n)
            
            if val < personal_best_vals[i]:
                personal_best_vals[i] = val
                personal_bests[i] = particles[i].copy()
                if val < best_global_val:
                    best_global_val = val
                    best_global = particles[i].copy()
        
        w = max(0.5, w * 0.998)
    
    return best_global, best_global_val

def _local_refine(inner_params, outer_angle_deg, n, time_limit=15.0):
    """Nelder-Mead local refinement."""
    start_time = time.time()
    init_params = inner_params.flatten()
    
    try:
        res = minimize(_objective, init_params, args=(outer_angle_deg, n),
                      method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if time.time() - start_time < time_limit:
            candidate = res.x.reshape(n, 3)
            cand_side = _compute_side(candidate, outer_angle_deg)
            cur_side = _compute_side(inner_params, outer_angle_deg)
            if cand_side < cur_side and _is_valid(candidate, cand_side, outer_angle_deg, min_sep=1e-9):
                return candidate
    except Exception:
        pass
    
    return inner_params

def _high_precision_polish(inner_params, outer_angle_deg, n, time_limit=15.0):
    """High-precision polish: smooth L-BFGS-B then tight Nelder-Mead."""
    start_time = time.time()
    result = inner_params
    
    # Phase 1: Smooth gradient-based
    try:
        res = minimize(_objective_smooth, result.flatten(), args=(outer_angle_deg, n),
                      method='L-BFGS-B',
                      options={'maxiter': 600, 'gtol': 1e-11, 'ftol': 1e-13})
        
        if time.time() - start_time < time_limit:
            candidate = res.x.reshape(n, 3)
            cand_side = _compute_side(candidate, outer_angle_deg)
            cur_side = _compute_side(result, outer_angle_deg)
            if cand_side < cur_side and _is_valid(candidate, cand_side, outer_angle_deg, min_sep=1e-9):
                result = candidate
    except Exception:
        pass
    
    # Phase 2: Tight Nelder-Mead
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        try:
            res = minimize(_objective, result.flatten(), args=(outer_angle_deg, n),
                          method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-13})
            
            if time.time() - start_time < time_limit:
                candidate = res.x.reshape(n, 3)
                cand_side = _compute_side(candidate, outer_angle_deg)
                cur_side = _compute_side(result, outer_angle_deg)
                if cand_side < cur_side and _is_valid(candidate, cand_side, outer_angle_deg, min_sep=1e-9):
                    result = candidate
        except Exception:
            pass
    
    return result

def _coordinate_descent_polish(params, outer_angle_deg, n, start_time, deadline, max_iter=5000, initial_step=0.01):
    """
    Coordinate descent polish: iterate over each parameter coordinate,
    try small perturbations, keep if improvement.
    """
    current = params.copy()
    current_score = _objective(current, outer_angle_deg, n)
    step_size = initial_step
    
    for iteration in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        improved = False
        for i in range(len(current)):
            if time.time() - start_time > deadline:
                break
            
            for direction in [1, -1]:
                trial = current.copy()
                trial[i] += direction * step_size
                trial_score = _objective(trial, outer_angle_deg, n)
                
                if trial_score < current_score - 1e-13:
                    current = trial
                    current_score = trial_score
                    improved = True
        
        if not improved:
            step_size *= 0.6
            if step_size < 1e-11:
                break
    
    return current, current_score

def construct():
    start_time = time.time()
    safety_margin = 5.0
    
    n = 11
    outer_angle = 30.0
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    best_side = seed_side
    
    # Phase 1: Morphing search (smooth gradient-based from lattice)
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 15:
        morph_time = min(remaining * 0.35, 40.0)
        try:
            morph_result, morph_val = _morphing_search(best_inner, outer_angle, n, time_limit=morph_time)
            if morph_val < best_side:
                best_inner = morph_result
                best_side = morph_val
        except Exception:
            pass
    
    # Phase 2: PSO for exploration
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        pso_time = min(remaining * 0.3, 25.0)
        try:
            pso_result, pso_val = _psp_swarm(best_inner, outer_angle, n, iterations=120, swarm_size=15, time_limit=pso_time)
            if pso_val < best_side:
                candidate = pso_result
                candidate_side = _compute_side(candidate, outer_angle)
                if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Phase 3: Local refinement
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            refined = _local_refine(best_inner, outer_angle, n, time_limit=min(remaining * 0.4, 12.0))
            refined_side = _compute_side(refined, outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass
    
    # Phase 4: High-precision polish
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 4:
        try:
            polished = _high_precision_polish(best_inner, outer_angle, n, time_limit=min(remaining * 0.5, 12.0))
            polished_side = _compute_side(polished, outer_angle)
            if polished_side < best_side and _is_valid(polished, polished_side, outer_angle, min_sep=1e-9):
                best_inner = polished
                best_side = polished_side
        except Exception:
            pass
    
    # Phase 5: Coordinate descent for final micro-optimization
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3:
        try:
            cd_params, cd_score = _coordinate_descent_polish(
                best_inner.flatten(), outer_angle, n, start_time,
                DEADLINE - safety_margin, max_iter=3000, initial_step=0.005)
            cd_candidate = cd_params.reshape(n, 3)
            cd_side = _compute_side(cd_candidate, outer_angle)
            if cd_side < best_side and _is_valid(cd_candidate, cd_side, outer_angle, min_sep=1e-9):
                best_inner = cd_candidate
                best_side = cd_side
        except Exception:
            pass
    
    # Final validation
    final_side = _compute_side(best_inner, outer_angle)
    if not _is_valid(best_inner, final_side, outer_angle, min_sep=1e-9):
        best_inner = np.array(seed_inner)
        final_side = seed_side
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(outer_angle)
    
    return inner_rows, center, side, angle_deg
