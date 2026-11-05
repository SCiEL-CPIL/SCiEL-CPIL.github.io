import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 145.0
SAFETY = 5.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _outer_hex_vertices(cx, cy, side, angle_deg):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
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
        t = np.clip(f / e, 0, 1)
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = np.clip(c / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = np.clip(s, 0, 1)
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = np.clip(c / a, 0, 1)
        elif t > 1:
            t = 1
            s = np.clip((b + f) / a, 0, 1)
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        s = 0
        t = np.clip(f / e, 0, 1)
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
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

def _compute_side(inner_params, outer_angle_deg, lo=1.5, hi=15.0):
    n = len(inner_params)
    for _ in range(50):
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

def _overlap_penalty(inner_params, penalty_weight=500.0):
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
                penalty += penalty_weight * (0.005 - sep) ** 2
    return penalty

def _objective(params, outer_angle_deg, n):
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
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
    eps = 1e-9
    centers = [(0.0, 0.0)]
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
    inner = [[x, y, 30.0] for x, y in centers]
    max_x = (2 + eps) * SQRT3
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / SQRT3
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    return inner, (0.0, 0.0), side, 30.0

def _golden_search(func, a, b, tol=1e-9, max_iter=200):
    phi = (math.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = func(c)
    fd = func(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = func(d)
    return (a + b) / 2

def _compute_side_for_rotation(inner_params, outer_angle_deg, n):
    return _compute_side(inner_params, outer_angle_deg)

def _search_outer_rotation(inner_params, n, start_time, base_angle=30.0):
    """1D golden-section search over outer hexagon rotation angle."""
    def obj(angle_deg):
        return _compute_side(inner_params, angle_deg)
    
    best_angle = base_angle
    best_side = obj(base_angle)
    
    # Scan a range of angles to find a good initial bracket
    angles = np.linspace(base_angle - 30, base_angle + 30, 61)
    for ang in angles:
        if time.time() - start_time > DEADLINE - SAFETY:
            break
        s = obj(ang)
        if s < best_side:
            best_side = s
            best_angle = ang
    
    # Golden section around the best found
    lo = best_angle - 5.0
    hi = best_angle + 5.0
    result_angle = _golden_search(lambda a: obj(a), lo, hi, tol=1e-7, max_iter=100)
    result_side = obj(result_angle)
    
    if result_side < best_side:
        best_angle = result_angle
        best_side = result_side
    
    return best_angle, best_side

def _psp_swarm(inner_params, outer_angle_deg, n, iterations=200, swarm_size=20, time_limit=30.0):
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
            particles[i] = inner_params + np.random.randn(n, 3) * 0.3
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)
        val = _objective(particles[i].flatten(), outer_angle_deg, n)
        personal_bests[i] = particles[i]
        personal_best_vals[i] = val
        if val < best_global_val:
            best_global_val = val
            best_global = particles[i].copy()
    
    w = 0.85
    c1 = 1.4
    c2 = 1.4
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        for i in range(swarm_size):
            r1 = np.random.rand(n, 3)
            r2 = np.random.rand(n, 3)
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_bests[i] - particles[i]) + c2 * r2 * (best_global - particles[i])
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
        w = max(0.4, w * 0.998)
    
    return best_global, best_global_val

def _local_refine(inner_params, outer_angle_deg, n, time_limit=15.0):
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

def _high_precision_polish(inner_params, outer_angle_deg, n):
    start_time = time.time()
    init_params = inner_params.flatten()
    try:
        res = minimize(_objective, init_params, args=(outer_angle_deg, n),
                      method='Nelder-Mead',
                      options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-11})
        if time.time() - start_time < 10.0:
            candidate = res.x.reshape(n, 3)
            if res.fun < _objective(inner_params.flatten(), outer_angle_deg, n):
                return candidate
    except Exception:
        pass
    return inner_params

def _multi_restart_optimize(inner_params, outer_angle_deg, n, start_time):
    """Multiple restarts with different initial perturbations."""
    best_inner = inner_params.copy()
    best_side = _compute_side(best_inner, outer_angle_deg)
    
    for restart in range(5):
        if time.time() - start_time > DEADLINE - SAFETY - 20:
            break
        
        # Create perturbed initial point
        perturbed = inner_params + np.random.randn(n, 3) * 0.2
        perturbed[:, 2] = np.mod(perturbed[:, 2], 360)
        
        try:
            res = minimize(_objective, perturbed.flatten(), args=(outer_angle_deg, n),
                          method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
            candidate = res.x.reshape(n, 3)
            candidate_side = _compute_side(candidate, outer_angle_deg)
            if candidate_side < best_side and _is_valid(candidate, candidate_side, outer_angle_deg, min_sep=1e-9):
                best_inner = candidate
                best_side = candidate_side
        except Exception:
            pass
    
    return best_inner, best_side

def construct():
    start_time = time.time()
    
    n = 11
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    best_side = seed_side
    outer_angle = 30.0
    
    # Phase 1: Initial Nelder-Mead refinement
    if time.time() - start_time < DEADLINE - SAFETY - 30:
        init_params = best_inner.flatten()
        try:
            res = minimize(_objective, init_params, args=(outer_angle, n),
                          method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_side:
                candidate = res.x.reshape(n, 3)
                candidate_side = _compute_side(candidate, outer_angle)
                if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Phase 2: Search over outer rotation angle
    if time.time() - start_time < DEADLINE - SAFETY - 25:
        new_angle, new_side = _search_outer_rotation(best_inner, n, start_time, base_angle=30.0)
        if new_side < best_side:
            outer_angle = new_angle
            best_side = new_side
            # Refine positions with new outer angle
            try:
                res = minimize(_objective, best_inner.flatten(), args=(outer_angle, n),
                              method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
                candidate = res.x.reshape(n, 3)
                candidate_side = _compute_side(candidate, outer_angle)
                if candidate_side < best_side and _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
            except Exception:
                pass
    
    # Phase 3: PSO optimization
    remaining = DEADLINE - SAFETY - (time.time() - start_time)
    if remaining > 15:
        pso_time = min(remaining * 0.4, 35.0)
        try:
            pso_result, pso_val = _psp_swarm(best_inner, outer_angle, n, iterations=150, swarm_size=20, time_limit=pso_time)
            candidate_side = _compute_side(pso_result, outer_angle)
            if candidate_side < best_side and _is_valid(pso_result, candidate_side, outer_angle, min_sep=1e-9):
                best_inner = pso_result
                best_side = candidate_side
        except Exception:
            pass
    
    # Phase 4: Multi-restart optimization
    if time.time() - start_time < DEADLINE - SAFETY - 10:
        refined_inner, refined_side = _multi_restart_optimize(best_inner, outer_angle, n, start_time)
        if refined_side < best_side:
            best_inner = refined_inner
            best_side = refined_side
    
    # Phase 5: Local refinement
    remaining = DEADLINE - SAFETY - (time.time() - start_time)
    if remaining > 5:
        try:
            refined = _local_refine(best_inner, outer_angle, n, time_limit=min(remaining * 0.5, 15.0))
            refined_side = _compute_side(refined, outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass
    
    # Phase 6: High-precision polish
    remaining = DEADLINE - SAFETY - (time.time() - start_time)
    if remaining > 3:
        try:
            polished = _high_precision_polish(best_inner, outer_angle, n)
            polished_side = _compute_side(polished, outer_angle)
            if polished_side < best_side and _is_valid(polished, polished_side, outer_angle, min_sep=1e-9):
                best_inner = polished
                best_side = polished_side
        except Exception:
            pass
    
    # Final validation
    final_side = _compute_side(best_inner, outer_angle)
    if not _is_valid(best_inner, final_side, outer_angle, min_sep=1e-9):
        best_inner = np.array(seed_inner)
        final_side = seed_side
        outer_angle = 30.0
    
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
    
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(outer_angle)
    
    return inner_rows, center, side, angle_deg
