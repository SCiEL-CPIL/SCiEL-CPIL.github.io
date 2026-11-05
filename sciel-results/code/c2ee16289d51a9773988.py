import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

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
    Returns 0 if they overlap.
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

def _compute_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon.
    inner_params: array of shape (n, 3) with [x, y, angle_deg]
    """
    n = len(inner_params)
    
    lo = 2.0
    hi = 12.0
    
    for _ in range(45):
        mid = (lo + hi) / 2
        outer_v = _outer_hex_vertices(0.0, 0.0, mid, outer_angle_deg)
        
        valid = True
        for i in range(n):
            x = inner_params[i, 0]
            y = inner_params[i, 1]
            a = inner_params[i, 2]
            hex_v = _hex_vertices(x, y, a)
            for v in hex_v:
                if not _point_in_polygon(v, outer_v):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            hi = mid
        else:
            lo = mid
    
    return hi

def _overlap_penalty(inner_params, penalty_weight=200.0):
    """
    Compute penalty for overlapping inner hexagons.
    """
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
    """
    Objective function: outer side length + overlap penalty.
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

def _psp_swarm(inner_params, outer_angle_deg, n, iterations=300, swarm_size=30, time_limit=30.0):
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
            particles[i] = inner_params + np.random.randn(n, 3) * 0.5
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)
        
        val = _objective(particles[i].flatten(), outer_angle_deg, n)
        personal_bests[i] = particles[i]
        personal_best_vals[i] = val
        
        if val < best_global_val:
            best_global_val = val
            best_global = particles[i].copy()
    
    # PSO parameters
    w = 0.9  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        for i in range(swarm_size):
            r1 = np.random.rand(n, 3)
            r2 = np.random.rand(n, 3)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_bests[i] - particles[i]) + c2 * r2 * (best_global - particles[i])
            
            # Clamp velocities
            max_vel = 0.5
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
        w = max(0.4, w * 0.999)
    
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
                      options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
        
        if time.time() - start_time < time_limit:
            candidate = res.x.reshape(n, 3)
            if res.fun < _objective(inner_params.flatten(), outer_angle_deg, n):
                return candidate
    except Exception:
        pass
    
    return inner_params

def _analytic_polish(inner_params, outer_angle_deg, n):
    """
    High-precision polish pass using the insight that the 'flower' packing
    (1 center, 6 inner ring, 4 outer ring) has a known optimal structure.
    We refine the positions of the 11 hexagons to minimize the outer side length
    by treating the problem as finding the tightest bounding hexagon.
    
    This function attempts to:
    1. Identify the structural roles of each hexagon (center, ring1, ring2).
    2. Optimize the radii of the rings and the angles to minimize the bounding hexagon side.
    3. Use a smooth surrogate for the max-projection objective.
    """
    start_time = time.time()
    
    # The optimal flower packing for 11 hexagons:
    # - 1 at center (0,0)
    # - 6 at distance d1 = sqrt(3) (touching center and each other)
    # - 4 at distance d2 = 3.0 (in the gaps of the first ring)
    
    # Let's parameterize by d1, d2, and the rotation of the outer hexagon.
    # Actually, the inner hexagons are all rotated by 30 deg relative to the outer.
    # Let's just refine the positions using a more targeted optimization.
    
    # We'll use a smooth max function (log-sum-exp) for the side length calculation
    # to allow gradient-based optimization if we were using that, but here we'll
    # just do a very tight Nelder-Mead on the final positions.
    
    # First, let's snap the structure to the known optimal topology if we're close.
    # If the configuration looks like a flower, we can enforce the symmetry.
    
    # For now, let's just do a very high-precision Nelder-Mead refinement.
    init_params = inner_params.flatten()
    
    try:
        res = minimize(_objective, init_params, args=(outer_angle_deg, n), 
                      method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if time.time() - start_time < 10.0:
            candidate = res.x.reshape(n, 3)
            if res.fun < _objective(inner_params.flatten(), outer_angle_deg, n):
                return candidate
    except Exception:
        pass
    
    return inner_params

def _fast_side_length(params, n, outer_angle=30.0):
    """
    Fast analytical computation of outer side length using projection onto 6 face normals.
    params: array of shape (n, 3) with [x, y, angle_deg]
    outer_angle: rotation of the outer hexagon in degrees
    Returns the minimum outer hexagon side length.
    """
    # Face normals of outer hexagon (pointing outward)
    # For a hexagon with vertices at angles outer_angle, outer_angle+60, ...,
    # the face normals are at angles outer_angle-30, outer_angle+30, ...
    # Actually, for a regular hexagon with vertices at angles theta, theta+60, ...,
    # the face normals are at angles theta-30, theta+30, theta+90, ...
    # Let me think: if vertices are at angles 0,60,120,180,240,300,
    # then faces are between consecutive vertices. The normal to the face between
    # vertices at 0 and 60 degrees points in the direction of the angle bisector,
    # which is 30 degrees. So normals are at 30, 90, 150, 210, 270, 330.
    # If we rotate the hexagon by outer_angle, normals are at outer_angle+30, outer_angle+90, ...
    normal_angles = np.deg2rad(outer_angle + np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for i in range(n):
        x = params[i, 0]
        y = params[i, 1]
        a = params[i, 2]
        verts = _hex_vertices(x, y, a)
        dots = verts @ normals.T
        max_proj = max(max_proj, np.max(np.abs(dots)))
    
    # For a regular hexagon with side S, apothem = S * sqrt(3) / 2
    # So S = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / SQRT3

def _smooth_objective(params, n, outer_angle=30.0, t=50.0):
    """
    Smooth approximation of the side length using log-sum-exp.
    params: array of shape (n, 3) with [x, y, angle_deg]
    outer_angle: rotation of the outer hexagon in degrees
    t: temperature parameter for log-sum-exp (higher = smoother)
    """
    normal_angles = np.deg2rad(outer_angle + np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    all_dots = []
    for i in range(n):
        x = params[i, 0]
        y = params[i, 1]
        a = params[i, 2]
        verts = _hex_vertices(x, y, a)
        dots = verts @ normals.T  # shape (6, 6)
        all_dots.append(np.abs(dots))
    
    all_dots = np.vstack(all_dots)  # shape (6n, 6)
    
    # For each normal direction, take max over all vertices
    max_per_normal = np.max(all_dots, axis=0)  # shape (6,)
    
    # Log-sum-exp approximation of max over the 6 normals
    m = np.max(max_per_normal)
    log_sum_exp = m + np.log(np.sum(np.exp(max_per_normal - m)))
    smooth_max = log_sum_exp / t
    
    return 2.0 * smooth_max / SQRT3

def _coordinate_descent_polish(inner_params, outer_angle_deg, n, time_limit=15.0):
    """
    Coordinate descent with fine-grained steps for high-precision polish.
    Also optimizes the outer hexagon angle.
    """
    start_time = time.time()
    params = inner_params.flatten().copy()
    oa = outer_angle_deg
    current_val = _fast_side_length(params.reshape(n, 3), n, oa)
    
    # Try small perturbations in each dimension
    step_sizes = [1e-3, 1e-4, 1e-5, 1e-6]
    
    for step in step_sizes:
        if time.time() - start_time > time_limit:
            break
        
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            improved = False
            for dim in range(len(params)):
                # Try positive and negative perturbations
                for direction in [1, -1]:
                    test_params = params.copy()
                    test_params[dim] += direction * step
                    test_val = _fast_side_length(test_params.reshape(n, 3), n, oa)
                    if test_val < current_val:
                        params = test_params
                        current_val = test_val
                        improved = True
            # Also try perturbing the outer angle
            for direction in [1, -1]:
                test_oa = oa + direction * step
                test_val = _fast_side_length(params.reshape(n, 3), n, test_oa)
                if test_val < current_val:
                    oa = test_oa
                    current_val = test_val
                    improved = True
    
    return params.reshape(n, 3), current_val, oa

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
    best_outer_angle = outer_angle
    
    # Phase 1: Fast analytical optimization using projection method
    # Use the fast side length computation for initial optimization
    init_params = best_inner.flatten()
    
    # Multi-start Powell optimization with fast objective
    n_starts = 8
    best_params = init_params.copy()
    best_val = _fast_side_length(best_params.reshape(n, 3), n, outer_angle)
    
    for i in range(n_starts):
        if time.time() > DEADLINE - safety_margin - 25:
            break
        
        if i == 0:
            x_start = init_params.copy()
        elif i == 1:
            x_start = init_params + np.random.randn(n, 3) * 0.1
        elif i == 2:
            x_start = init_params + np.random.randn(n, 3) * 0.3
        elif i == 3:
            x_start = init_params + np.random.randn(n, 3) * 0.5
        elif i == 4:
            x_start = init_params + np.random.randn(n, 3) * 0.8
        else:
            x_start = init_params + np.random.randn(n, 3) * 1.2
        
        x_start[:, 2] = np.mod(x_start[:, 2], 360)
        
        try:
            res = minimize(lambda p: _fast_side_length(p.reshape(n, 3), n, outer_angle), x_start, 
                          method='Powell',
                          options={'maxiter': 800, 'disp': False, 'ftol': 1e-13, 'xtol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_val < best_side:
        candidate = best_params.reshape(n, 3)
        candidate_side = _compute_side(candidate, outer_angle)
        if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
            best_inner = candidate
            best_side = candidate_side
    
    # Phase 1b: Optimize outer hexagon angle using coordinate descent
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 8:
        try:
            # Try different outer angles
            for test_oa in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]:
                if time.time() > DEADLINE - safety_margin - 3:
                    break
                test_side = _fast_side_length(best_inner, n, test_oa)
                if test_side < best_side:
                    # Verify with exact computation
                    exact_side = _compute_side(best_inner, test_oa)
                    if exact_side < best_side and _is_valid(best_inner, exact_side, test_oa, min_sep=1e-9):
                        best_side = exact_side
                        best_outer_angle = test_oa
        except Exception:
            pass
    
    # Phase 2: PSO with the original objective (includes overlap penalty)
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        pso_time = min(remaining * 0.35, 25.0)
        try:
            pso_result, pso_val = _psp_swarm(best_inner, best_outer_angle, n, iterations=120, swarm_size=20, time_limit=pso_time)
            if pso_val < best_side:
                candidate = pso_result
                candidate_side = _compute_side(candidate, best_outer_angle)
                if _is_valid(candidate, candidate_side, best_outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Phase 3: Local refinement with Nelder-Mead
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            refined = _local_refine(best_inner, best_outer_angle, n, time_limit=min(remaining * 0.35, 12.0))
            refined_side = _compute_side(refined, best_outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, best_outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass
    
    # Phase 4: Coordinate descent polish with fast objective (also optimizes outer angle)
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 4:
        try:
            polished, polished_val, polished_oa = _coordinate_descent_polish(best_inner, best_outer_angle, n, time_limit=min(remaining * 0.5, 10.0))
            polished_side = _compute_side(polished, polished_oa)
            if polished_side < best_side and _is_valid(polished, polished_side, polished_oa, min_sep=1e-9):
                best_inner = polished
                best_side = polished_side
                best_outer_angle = polished_oa
        except Exception:
            pass
    
    # Phase 5: Final high-precision Nelder-Mead polish
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3:
        try:
            init_params = best_inner.flatten()
            res = minimize(_objective, init_params, args=(best_outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12})
            if time.time() - start_time < DEADLINE - safety_margin:
                candidate = res.x.reshape(n, 3)
                candidate_side = _compute_side(candidate, best_outer_angle)
                if candidate_side < best_side and _is_valid(candidate, candidate_side, best_outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Phase 6: Final outer angle optimization
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 2:
        try:
            # Fine-tune outer angle
            for delta in [-0.5, -0.1, 0.0, 0.1, 0.5]:
                if time.time() > DEADLINE - safety_margin - 1:
                    break
                test_oa = best_outer_angle + delta
                test_side = _fast_side_length(best_inner, n, test_oa)
                if test_side < best_side:
                    exact_side = _compute_side(best_inner, test_oa)
                    if exact_side < best_side and _is_valid(best_inner, exact_side, test_oa, min_sep=1e-9):
                        best_side = exact_side
                        best_outer_angle = test_oa
        except Exception:
            pass
    
    # Final check
    final_side = _compute_side(best_inner, best_outer_angle)
    if not _is_valid(best_inner, final_side, best_outer_angle, min_sep=1e-9):
        # Fallback to seed
        best_inner = np.array(seed_inner)
        final_side = seed_side
        best_outer_angle = 30.0
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(best_outer_angle)
    
    return inner_rows, center, side, angle_deg
