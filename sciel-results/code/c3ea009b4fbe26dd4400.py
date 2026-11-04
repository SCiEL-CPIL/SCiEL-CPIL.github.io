"""Packing 11 unit hexagons in a regular hexagon - morphing lattice approach."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

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
    Returns 0 if they overlap, positive if separated.
    """
    # Check if any vertex of p1 is inside p2
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    # Check if any vertex of p2 is inside p1
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    # Check edge-edge distances
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
    are contained within the outer hexagon centered at origin.
    inner_params: array of shape (n, 3) with [x, y, angle_deg]
    """
    n = len(inner_params)
    
    # Binary search on side length
    lo = 2.0
    hi = 12.0
    
    for _ in range(40):
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

def _overlap_penalty(inner_params, penalty_weight=50.0):
    """
    Compute penalty for overlapping inner hexagons.
    Returns 0 if all hexagons are disjoint, positive otherwise.
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
    params: flat array of length 3*n: [x1, y1, a1, x2, y2, a2, ...]
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    # Check containment
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    # Check non-overlap
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
    """Return the concentric ring seed solution with scaling for strict disjointness."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    # 6 hexagons in first ring at distance sqrt(3) (touching center)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    # 4 more hexagons in second ring
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # Compute bounding hexagon side
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def construct():
    """Main entry point for packing 11 unit hexagons."""
    start_time = time.time()
    DEADLINE = 140.0
    
    n = 11
    outer_angle = 30.0
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner
    best_side = seed_side
    best_valid = True
    
    # Initial guess: concentric ring
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(seed_inner):
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = a
    
    best_score = seed_side
    best_params = init_params.copy()
    
    # Try multiple optimization approaches
    attempts = [
        ('L-BFGS-B', 'lbfgs'),
        ('Nelder-Mead', 'nelder'),
    ]
    
    for opt_name, method in attempts:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        try:
            result = minimize(
                _objective,
                init_params,
                args=(outer_angle, n),
                method=method,
                options={'maxiter': 200, 'maxfev': 5000}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Try differential evolution with smaller population for speed
    if time.time() - start_time < DEADLINE - 30:
        try:
            bounds = [(-6, 6), (-6, 6), (-30, 30)] * n
            de_result = differential_evolution(
                _objective,
                bounds,
                args=(outer_angle, n),
                maxiter=50,
                popsize=8,
                tol=1e-4,
                seed=42,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B if time permits
    if time.time() - start_time < DEADLINE - 10:
        try:
            polish_result = minimize(
                _objective,
                best_params,
                args=(outer_angle, n),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfev': 2000}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_params = polish_result.x.copy()
        except Exception:
            pass
    
    # Extract final solution
    inner_params = best_params.reshape(n, 3)
    
    # Center the configuration
    all_verts = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        v = _hex_vertices(x, y, a)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    centroid = np.mean(all_verts, axis=0)
    
    # Shift to center
    inner = []
    for i in range(n):
        x = inner_params[i, 0] - centroid[0]
        y = inner_params[i, 1] - centroid[1]
        a = inner_params[i, 2]
        inner.append([x, y, a])
    
    # Recompute side length with centered configuration
    centered_params = np.array(inner).reshape(n, 3)
    side = _compute_side(centered_params, outer_angle)
    
    # Verify validity, if invalid use seed
    if not _is_valid(centered_params, side, outer_angle, min_sep=-1e-6):
        inner = seed_inner
        side = seed_side
        outer_angle = seed_angle
    
    return inner, (0.0, 0.0), side, outer_angle
