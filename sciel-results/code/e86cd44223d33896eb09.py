import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def _soft_max(vals, beta=50.0):
    """Smooth approximation of max via log-sum-exp."""
    m = np.max(vals)
    if m < -1e10:
        return m
    return m + np.log(np.sum(np.exp(vals - m))) / beta

def _support_function_smooth(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length using smooth support function.
    For a regular hexagon with side s centered at origin rotated by outer_angle,
    a point p is inside iff max over 6 face normals of dot(normal_i, p) <= s * apothem.
    Actually for regular hexagon with side s, the inradius (apothem) = s * sqrt(3)/2.
    The 6 face normals are at angles outer_angle + 30 + k*60 degrees.
    A point p is inside iff for all k: dot(n_k, p) <= apothem = s * sqrt(3)/2.
    So s >= max_k(dot(n_k, p)) / (sqrt(3)/2) for all points p in all inner hexagons.
    We use the vertices of inner hexagons as representative points.
    """
    n = len(inner_params)
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    # Face normals of outer hexagon
    normal_angles = outer_angle_rad + np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # shape (6, 2)
    
    # Collect all vertices of all inner hexagons
    all_pts = []
    for i in range(n):
        x, y, a = inner_params[i]
        vs = _hex_vertices(x, y, a)
        all_pts.append(vs)
    all_pts = np.vstack(all_pts)  # shape (6n, 2)
    
    # For each normal, compute max dot product over all points
    # dots: shape (6, 6n)
    dots = normals @ all_pts.T  # (6, 6n)
    max_dots = np.max(dots, axis=1)  # (6,)
    
    # Smooth max over the 6 face constraints
    sm = _soft_max(max_dots, beta=100.0)
    
    # side = sm / (sqrt(3)/2)
    return sm / (SQRT3 / 2.0)

def _objective_smooth(params, outer_angle_deg, n):
    """Smooth objective: soft support function + overlap penalty."""
    inner_params = params.reshape(n, 3)
    side = _support_function_smooth(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

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
    """Compute the minimum distance between two convex polygons. Returns 0 if they overlap."""
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

def _compute_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon.
    Uses exact vertex containment check via support function (max dot product).
    """
    n = len(inner_params)
    
    # Precompute all inner hex vertices
    all_pts = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        all_pts.append(hex_v)
    all_pts = np.vstack(all_pts)
    
    # Face normals of outer hexagon
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle_rad + np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each normal, max dot product over all vertices
    dots = normals @ all_pts.T  # (6, 6n)
    max_dots = np.max(dots, axis=1)  # (6,)
    
    # side = max(max_dots) / (sqrt(3)/2)
    return np.max(max_dots) / (SQRT3 / 2.0)

def _overlap_penalty(inner_params, penalty_weight=100.0):
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
            if sep < 0.01:
                penalty += penalty_weight * (0.01 - sep) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n):
    """Objective function: outer side length + overlap penalty."""
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _objective_gradient(params, outer_angle_deg, n):
    """
    Objective with analytic gradient for the support function part.
    The support function is piecewise linear; we use subgradient.
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid."""
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

def _psp_swarm(inner_params, outer_angle_deg, n, iterations=200, swarm_size=20, time_limit=30.0):
    """
    Particle Swarm Optimization to minimize the objective.
    Returns the best particle found.
    """
    start_time = time.time()
    best_global = inner_params.copy()
    best_global_val = _objective(best_global.flatten(), outer_angle_deg, n)
    
    # Initialize swarm
    particles = np.zeros((swarm_size, n, 3))
    velocities = np.zeros((swarm_size, n, 3))
    personal_bests = np.zeros((swarm_size, n, 3))
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        if i == 0:
            particles[i] = inner_params
        else:
            particles[i] = inner_params + np.random.randn(n, 3) * 0.4
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)
        
        val = _objective(particles[i].flatten(), outer_angle_deg, n)
        personal_bests[i] = particles[i]
        personal_best_vals[i] = val
        
        if val < best_global_val:
            best_global_val = val
            best_global = particles[i].copy()
    
    # PSO parameters
    w = 0.85  # inertia
    c1 = 1.2  # cognitive
    c2 = 1.2  # social
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        for i in range(swarm_size):
            r1 = np.random.rand(n, 3)
            r2 = np.random.rand(n, 3)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_bests[i] - particles[i]) + c2 * r2 * (best_global - particles[i])
            
            # Clamp velocities
            max_vel = 0.3
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
            
            particles[i] = particles[i] + velocities[i]
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)
            
            val = _objective(particles[i].flatten(), outer_angle_deg, n)
            
            if val < personal_best_vals[i]:
                personal_best_vals[i] = val
                personal_bests[i] = particles[i].copy()
                
                if val < best_global_val:
                    best_global_val = val
                    best_global = particles[i].copy()
        
        # Decay inertia
        w = max(0.5, w * 0.998)
    
    return best_global, best_global_val

def _local_refine(inner_params, outer_angle_deg, n, time_limit=20.0):
    """
    Use Nelder-Mead for local refinement.
    """
    start_time = time.time()
    
    init_params = inner_params.flatten()
    
    try:
        res = minimize(_objective, init_params, args=(outer_angle_deg, n), 
                      method='Nelder-Mead', 
                      options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if time.time() - start_time < time_limit:
            candidate = res.x.reshape(n, 3)
            if res.fun < _objective(inner_params.flatten(), outer_angle_deg, n):
                return candidate
    except Exception:
        pass
    
    return inner_params

def _high_precision_polish(inner_params, outer_angle_deg, n, time_limit=15.0):
    """
    High-precision polish: first use smooth objective for gradient-based refinement,
    then exact Nelder-Mead with very tight tolerances.
    """
    start_time = time.time()
    result = inner_params
    
    # Phase 1: Smooth objective with L-BFGS-B for gradient-based improvement
    init_params = result.flatten()
    try:
        res = minimize(_objective_smooth, init_params, args=(outer_angle_deg, n),
                      method='L-BFGS-B',
                      options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
        
        if time.time() - start_time < time_limit:
            candidate = res.x.reshape(n, 3)
            cand_side = _compute_side(candidate, outer_angle_deg)
            cur_side = _compute_side(result, outer_angle_deg)
            if cand_side < cur_side and _is_valid(candidate, cand_side, outer_angle_deg, min_sep=1e-9):
                result = candidate
    except Exception:
        pass
    
    # Phase 2: Exact Nelder-Mead with very tight tolerances
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        init_params = result.flatten()
        try:
            res = minimize(_objective, init_params, args=(outer_angle_deg, n),
                          method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if time.time() - start_time < time_limit:
                candidate = res.x.reshape(n, 3)
                cand_side = _compute_side(candidate, outer_angle_deg)
                cur_side = _compute_side(result, outer_angle_deg)
                if cand_side < cur_side and _is_valid(candidate, cand_side, outer_angle_deg, min_sep=1e-9):
                    result = candidate
        except Exception:
            pass
    
    return result

def _morphing_search(inner_params, outer_angle_deg, n, time_limit=40.0):
    """
    Geometric morphing: parameterize inner hexagons as displacements from the
    hexagonal lattice, optimize with smooth objective + L-BFGS-B.
    """
    start_time = time.time()
    
    # Lattice positions (hexagonal packing)
    lattice_positions = np.array([
        [0.0, 0.0],
        [SQRT3, 0.0], [-SQRT3, 0.0],
        [SQRT3/2, 1.5], [-SQRT3/2, 1.5],
        [SQRT3/2, -1.5], [-SQRT3/2, -1.5],
        [2.0*SQRT3/2, 3.0], [-2.0*SQRT3/2, 3.0],
        [2.0*SQRT3/2, -3.0], [-2.0*SQRT3/2, -3.0],
    ])
    
    # Initial displacements = current - lattice (to start from current config)
    # But we'll also try starting from zero displacement
    best_result = inner_params
    best_val = _compute_side(inner_params, outer_angle_deg)
    
    # Try multiple initializations: zero displacement and current
    inits = [
        lattice_positions.copy(),
        inner_params.copy(),
    ]
    
    for init_idx, init in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
        
        init_params = init.flatten()
        
        # L-BFGS-B with smooth objective
        try:
            res = minimize(_objective_smooth, init_params, args=(outer_angle_deg, n),
                          method='L-BFGS-B',
                          options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10})
            
            if time.time() - start_time > time_limit:
                break
            
            candidate = res.x.reshape(n, 3)
            cand_side = _compute_side(candidate, outer_angle_deg)
            if cand_side < best_val and _is_valid(candidate, cand_side, outer_angle_deg, min_sep=1e-9):
                best_result = candidate
                best_val = cand_side
        except Exception:
            pass
        
        # Follow up with Nelder-Mead on the best smooth result
        if time.time() - start_time > time_limit - 5:
            break
        
        try:
            res2 = minimize(_objective, init_params, args=(outer_angle_deg, n),
                           method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9})
            
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

def construct():
    start_time = time.time()
    DEADLINE = 145.0
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
        morph_time = min(remaining * 0.4, 45.0)
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
        pso_time = min(remaining * 0.4, 30.0)
        try:
            pso_result, pso_val = _psp_swarm(best_inner, outer_angle, n, iterations=150, swarm_size=20, time_limit=pso_time)
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
            refined = _local_refine(best_inner, outer_angle, n, time_limit=min(remaining * 0.5, 15.0))
            refined_side = _compute_side(refined, outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass
    
    # Phase 4: High-precision polish (smooth + exact)
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3:
        try:
            polished = _high_precision_polish(best_inner, outer_angle, n, time_limit=min(remaining, 15.0))
            polished_side = _compute_side(polished, outer_angle)
            if polished_side < best_side and _is_valid(polished, polished_side, outer_angle, min_sep=1e-9):
                best_inner = polished
                best_side = polished_side
        except Exception:
            pass
    
    # Final check
    final_side = _compute_side(best_inner, outer_angle)
    if not _is_valid(best_inner, final_side, outer_angle, min_sep=1e-9):
        # Fallback to seed
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
