import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _hex_vertex_offsets(angle_deg, s=1.0):
    """Vertex offsets (relative to center) for a unit hexagon at given angle."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    return np.column_stack([s * np.cos(angles), s * np.sin(angles)])

def _outer_normals(angle_deg):
    """Outward unit normals for the 6 edges of a regular hexagon."""
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + angle_deg)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _convex_hull_2d(points):
    """Compute convex hull of 2D points using Andrew's monotone chain."""
    points = np.array(points, dtype=float)
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    
    return np.vstack([lower[:-1], upper[:-1]])

def _all_inner_vertices(inner_params):
    """Get all vertices of all inner hexagons."""
    verts = []
    for i in range(len(inner_params)):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        offsets = _hex_vertex_offsets(a)
        verts.append(offsets + np.array([x, y]))
    return np.vstack(verts)

def _compute_side_from_hull(hull_points, outer_angle_deg):
    """Compute minimum outer hexagon side given convex hull and outer angle.
    
    For a regular hexagon centered at origin with side S and rotation theta,
    constraint: for each outward normal n_k, n_k . p <= S * sqrt(3)/2 for all p in polygon.
    So S >= max_k max_p (n_k . p) / (sqrt(3)/2)
    """
    normals = _outer_normals(outer_angle_deg)
    cos30 = math.sqrt(3) / 2.0
    # Project all hull points onto all normals
    projs = hull_points @ normals.T  # (n_points, 6)
    max_proj = np.max(projs)
    return max_proj / cos30

def _min_enclosing_side_1d(inner_params, n_angles=720):
    """Minimize enclosing hexagon side over outer rotation angle using piecewise-linear structure.
    
    The function S(theta) = max_k max_p (n_k(theta) . p) / cos30 is piecewise-linear in theta.
    We sample densely and then refine with golden-section search.
    """
    hull = _convex_hull_2d(_all_inner_vertices(inner_params))
    
    def S_of_theta(theta_deg):
        return _compute_side_from_hull(hull, theta_deg)
    
    # First: coarse scan to find the minimum region
    thetas = np.linspace(0, 60, n_angles + 1)  # Period is 60 degrees
    S_vals = np.array([S_of_theta(t) for t in thetas])
    
    # Find minimum region
    min_idx = np.argmin(S_vals)
    
    # Refine with golden-section search around the minimum
    lo = thetas[max(0, min_idx - 2)]
    hi = thetas[min(n_angles, min_idx + 2)]
    
    # Golden section search
    gr = (math.sqrt(5) - 1) / 2  # golden ratio conjugate
    a = lo
    b = hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = S_of_theta(c)
    fd = S_of_theta(d)
    
    for _ in range(60):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = S_of_theta(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = S_of_theta(d)
        if abs(b - a) < 1e-10:
            break
    
    best_theta = (a + b) / 2
    best_S = S_of_theta(best_theta)
    
    return best_S, best_theta

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a <= 1e-12 and e <= 1e-12:
        return np.linalg.norm(r)
    
    if a <= 1e-12:
        t = f / e if e > 1e-12 else 0.0
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(r + t * d2)
    
    c = np.dot(d1, r)
    if e <= 1e-12:
        t = -c / a if a > 1e-12 else 0.0
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(d1 * t - r)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = (b * f - c * e) / denom if denom > 1e-12 else 0.0
    t = (a * f - b * c) / denom if denom > 1e-12 else 0.0
    
    if s < 0:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
    elif s > 1:
        s = 1.0
        t = (f - b) / e
        t = max(0.0, min(1.0, t))
    elif t < 0:
        t = 0.0
        s = -c / a
        s = max(0.0, min(1.0, s))
    elif t > 1:
        t = 1.0
        s = (f - b) / a
        s = max(0.0, min(1.0, s))
    
    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    return np.linalg.norm(closest1 - closest2)

def _poly_separation(v1, v2):
    """Minimum separation distance between two convex polygons (negative if overlapping)."""
    min_dist = np.inf
    n1, n2 = len(v1), len(v2)
    
    # Check all edge pairs
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            p3 = v2[j]
            p4 = v2[(j + 1) % n2]
            d = _seg_seg_dist(p1, p2, p3, p4)
            if d < min_dist:
                min_dist = d
    
    # Also check vertex-vertex distances for robustness
    diff = v1[:, np.newaxis, :] - v2[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    min_dist = min(min_dist, np.min(dists))
    
    # If polygons are overlapping, min_dist will be 0 from edge intersection
    # We need to detect actual overlap by checking if any vertex is inside the other
    for v in v1:
        if _point_in_polygon(v, v2):
            return -0.01  # Overlapping
    for v in v2:
        if _point_in_polygon(v, v1):
            return -0.01  # Overlapping
    
    return min_dist

def _smooth_overlap_penalty(inner_params, beta=100.0, gap=0.05):
    """Smooth penalty for overlapping inner hexagons using polygon separation."""
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        
        for j in range(i + 1, n):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            
            sep = _poly_separation(v1, v2)
            
            if sep < gap:
                penalty += (gap - sep) ** 2
    
    return beta * penalty

def _soft_max(vals, temperature=0.01):
    """Soft maximum using log-sum-exp for smoothness."""
    max_val = np.max(vals)
    return max_val + temperature * np.log(np.sum(np.exp((vals - max_val) / temperature)))

def _objective_with_min_side(params, n, beta=100.0):
    """Objective: min enclosing side + smooth overlap penalty."""
    inner_params = params.reshape(n, 3)
    
    # Compute min enclosing side via 1D optimization over outer angle
    hull = _convex_hull_2d(_all_inner_vertices(inner_params))
    
    # Use a denser angle scan for better accuracy
    thetas = np.linspace(0, 60, 241)
    
    # Vectorized computation of S(theta) for all thetas
    cos30 = math.sqrt(3) / 2.0
    S_vals = np.zeros(len(thetas))
    for k, theta in enumerate(thetas):
        normals = _outer_normals(theta)
        projs = hull @ normals.T
        S_vals[k] = np.max(projs) / cos30
    
    min_S = np.min(S_vals)
    
    # Refine around the minimum with more iterations
    min_idx = np.argmin(S_vals)
    lo = thetas[max(0, min_idx - 2)]
    hi = thetas[min(len(thetas) - 1, min_idx + 2)]
    
    # Golden section search with more iterations
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    def S_fn(theta):
        normals = _outer_normals(theta)
        projs = hull @ normals.T
        return np.max(projs) / cos30
    
    fc, fd = S_fn(c), S_fn(d)
    for _ in range(50):
        if fc < fd:
            b = d; d = c; fd = fc
            c = b - gr * (b - a); fc = S_fn(c)
        else:
            a = c; c = d; fc = fd
            d = a + gr * (b - a); fd = S_fn(d)
        if abs(b - a) < 1e-10:
            break
    
    min_S = min(min_S, S_fn((a + b) / 2))
    
    # Add overlap penalty
    pen = _smooth_overlap_penalty(inner_params, beta=beta)
    
    return min_S + pen

def _is_valid(inner_params, outer_angle_deg, tol=1e-6):
    """Check validity: containment and non-overlap."""
    n = len(inner_params)
    hull = _convex_hull_2d(_all_inner_vertices(inner_params))
    side = _compute_side_from_hull(hull, outer_angle_deg)
    
    # Check containment
    normals = _outer_normals(outer_angle_deg)
    cos30 = math.sqrt(3) / 2.0
    all_verts = _all_inner_vertices(inner_params)
    for nk in normals:
        projs = all_verts @ nk
        if np.max(projs) > side * cos30 + tol:
            return False
    
    # Check overlaps
    for i in range(n):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i + 1, n):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            # Check vertex-in-polygon
            for v in v1:
                if _point_in_polygon(v, v2):
                    return False
            for v in v2:
                if _point_in_polygon(v, v1):
                    return False
            # Check edge-edge intersections
            for k in range(6):
                a1 = v1[k]
                a2 = v1[(k + 1) % 6]
                for m in range(6):
                    b1 = v2[m]
                    b2 = v2[(m + 1) % 6]
                    if _segments_intersect(a1, a2, b1, b2):
                        return False
    return True

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
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

def _segments_intersect(p1, p2, p3, p4):
    """Check if segments p1-p2 and p3-p4 intersect."""
    def orient(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else -1
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    
    # Collinear cases
    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    if o4 == 0 and _on_segment(p3, p2, p4):
        return True
    
    return False

def _on_segment(a, b, c):
    """Check if b is on segment ac (collinear)."""
    return (min(a[0], c[0]) - 1e-12 <= b[0] <= max(a[0], c[0]) + 1e-12 and
            min(a[1], c[1]) - 1e-12 <= b[1] <= max(a[1], c[1]) + 1e-12)

def _seed_solution():
    """Return the concentric ring seed solution with optimized angles."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (D + eps) * np.cos(angle)
        y = (D + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2 * eps) * np.cos(angle)
        y = (3.0 + 2 * eps) * np.sin(angle)
        centers.append((x, y))
    
    # Try multiple angle configurations to find the best
    best_inner = None
    best_side = np.inf
    
    for base_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
        inner = []
        for x, y in centers:
            inner.append([x, y, base_angle])
        
        inner_params = np.array(inner)
        hull = _convex_hull_2d(_all_inner_vertices(inner_params))
        side, theta = _min_enclosing_side_1d(inner_params, n_angles=720)
        
        if side < best_side:
            best_side = side
            best_inner = inner
    
    return best_inner, (0.0, 0.0), best_side, theta

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    safety_margin = 15.0

    n = 11

    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner
    best_side = seed_side
    best_angle = seed_angle

    # Build initial parameter vector from seed
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(seed_inner):
        init_params[i * 3] = x
        init_params[i * 3 + 1] = y
        init_params[i * 3 + 2] = a

    # Multi-start optimization with diverse configurations
    rng = np.random.default_rng(42)
    starts = [init_params.copy()]
    
    # Add diverse starts: small perturbations
    for _ in range(6):
        pert = init_params.copy()
        pert[0::3] += rng.normal(0, 0.3, n)
        pert[1::3] += rng.normal(0, 0.3, n)
        pert[2::3] += rng.normal(0, 10, n)
        starts.append(pert)
    
    # Try different symmetric angle patterns
    for base_angle in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]:
        pert = init_params.copy()
        pert[2::3] = base_angle
        starts.append(pert)
    
    # Mixed angle patterns
    for pattern in [
        [0, 30, 60, 0, 30, 60, 0, 30, 60, 0, 30],
        [15, 45, 75, 15, 45, 75, 15, 45, 75, 15, 45],
        [0, 0, 0, 30, 30, 30, 60, 60, 60, 0, 30]
    ]:
        pert = init_params.copy()
        pert[2::3] = pattern
        starts.append(pert)

    for start_idx, start_params in enumerate(starts):
        if time.time() > DEADLINE - safety_margin:
            break
        
        # Nelder-Mead on the smooth objective
        try:
            res = minimize(_objective_with_min_side, start_params, args=(n,),
                           method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_side:
                inner_params = res.x.reshape(n, 3)
                # Find optimal outer angle
                hull = _convex_hull_2d(_all_inner_vertices(inner_params))
                S, theta = _min_enclosing_side_1d(inner_params, n_angles=1440)
                if S < best_side:
                    best_inner = [list(inner_params[i]) for i in range(n)]
                    best_side = S
                    best_angle = theta
        except Exception:
            pass
        
        # Additional polish with tighter settings if time permits
        if time.time() > DEADLINE - safety_margin:
            break
        try:
            # Use the best result so far for polish
            polish_start = res.x if 'res' in dir() else start_params
            res2 = minimize(_objective_with_min_side, polish_start, args=(n,),
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            
            if res2.fun < best_side:
                inner_params = res2.x.reshape(n, 3)
                hull = _convex_hull_2d(_all_inner_vertices(inner_params))
                S, theta = _min_enclosing_side_1d(inner_params, n_angles=1440)
                if S < best_side:
                    best_inner = [list(inner_params[i]) for i in range(n)]
                    best_side = S
                    best_angle = theta
        except Exception:
            pass

    # Final validation and precise side computation
    if best_inner:
        inner_params = np.array(best_inner)
        hull = _convex_hull_2d(_all_inner_vertices(inner_params))
        side, theta = _min_enclosing_side_1d(inner_params, n_angles=2880)
        
        # Validate
        if _is_valid(inner_params, theta, tol=1e-5):
            best_side = side
            best_angle = theta
        else:
            # Fall back to seed
            best_inner = seed_inner
            best_side = seed_side
            best_angle = seed_angle

    return best_inner, (0.0, 0.0), best_side, best_angle
