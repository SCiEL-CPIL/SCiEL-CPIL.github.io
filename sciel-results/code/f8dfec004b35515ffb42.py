import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
START = time.time()
DEADLINE = 148.0

def _hex_vertices(cx, cy, s, rot):
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    max_proj = 0.0
    for (x, y, a) in inner:
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / math.sqrt(3.0)

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_poly(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i+1) % n], pt)
        if abs(c) < 1e-9:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def _poly_overlap(A, B):
    """SAT: return True if convex polygons A and B overlap (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            if maxA < minB - 1e-7 or maxB < minA - 1e-7:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _hexagonal_seed():
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0) * 1.01
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _cluster_points():
    """Generate all 66 vertices of the fixed hexagonal cluster (1 center + 6 ring + 4 outer)."""
    points = []
    d = math.sqrt(3.0) * 1.01
    # Center hexagon vertices
    for k in range(6):
        a = math.radians(60 * k)
        points.append((math.cos(a), math.sin(a)))
    # Ring hexagons
    for i in range(6):
        base_a = math.radians(30 + i * 60)
        cx = d * math.cos(base_a)
        cy = d * math.sin(base_a)
        for k in range(6):
            a = base_a + math.radians(60 * k)
            points.append((cx + math.cos(a), cy + math.sin(a)))
    # Outer 4 hexagons
    d2 = 2 * d
    for i in range(4):
        base_a = math.radians(30 + i * 60)
        cx = d2 * math.cos(base_a)
        cy = d2 * math.sin(base_a)
        for k in range(6):
            a = base_a + math.radians(60 * k)
            points.append((cx + math.cos(a), cy + math.sin(a)))
    return np.array(points)

def _min_side_for_rotation(rot_deg):
    """Compute minimum enclosing hexagon side length for given rotation angle."""
    pts = _cluster_points()
    theta_rad = math.radians(rot_deg)
    normals = np.array([
        [math.cos(theta_rad + math.radians(30 + k * 60)),
         math.sin(theta_rad + math.radians(30 + k * 60))]
        for k in range(6)
    ])
    
    # Project all points onto all 6 normals
    projections = pts @ normals.T  # shape (66, 6)
    max_proj = np.max(projections)
    
    return 2.0 * max_proj / math.sqrt(3.0)

def _golden_section_search(f, a, b, tol=1e-10, max_iter=100):
    """Golden section search for minimum of unimodal function f on [a, b]."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    
    for _ in range(max_iter):
        if time.time() > START + DEADLINE - 5:
            break
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    
    return (a + b) / 2.0

def _objective(params):
    """
    Objective for optimization.
    params: [x1,y1,a1, ..., x11,y11,a11, ox, oy, rot]
    """
    pts = params[:N*3].reshape(N, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    thetas = pts[:, 2]
    ox, oy = params[N*3], params[N*3+1]
    rot = params[N*3+2]
    
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    max_proj = 0.0
    for k in range(N):
        a0 = math.radians(thetas[k])
        for m in range(6):
            a = a0 + math.radians(60 * m)
            px = xs[k] + math.cos(a)
            py = ys[k] + math.sin(a)
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    side = 2.0 * max_proj / math.sqrt(3.0)
    
    penalty = 0.0
    min_dist = math.sqrt(3.0)
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff * 1000.0
    
    outer_s = side
    for k in range(N):
        a0 = math.radians(thetas[k])
        for m in range(6):
            a = a0 + math.radians(60 * m)
            px = xs[k] + math.cos(a)
            py = ys[k] + math.sin(a)
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                limit = outer_s * math.sqrt(3.0) / 2.0
                if proj > limit:
                    penalty += (proj - limit) ** 2 * 1000.0
    
    return side + penalty

def _refine(inner, oc, rot, rng, budget):
    init_params = np.array([p for (x, y, a) in inner for p in [x, y, a]] + [oc[0], oc[1], rot])
    
    try:
        res_nm = minimize(
            _objective,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res_nm.fun < float('inf'):
            params = res_nm.x
            pts = params[:N*3].reshape(N, 3)
            inner_nm = [[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])] for i in range(N)]
            oc_nm = (float(params[N*3]), float(params[N*3+1]))
            rot_nm = float(params[N*3+2])
            s_nm = _min_outer_side(inner_nm, oc_nm, rot_nm)
            
            if _valid(inner_nm, oc_nm, s_nm, rot_nm):
                return inner_nm, oc_nm, s_nm, rot_nm
    except Exception:
        pass
    
    return None

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 10.0
    rng = random.Random(42)
    np.random.seed(42)
    
    # Phase 1: 1D golden-section search over rotation angle for fixed cluster
    # The cluster has 6-fold symmetry, so we only need to search [0, 30) degrees
    # but due to the asymmetric outer ring (4 hexagons instead of 6), the period is 60°
    
    # Coarse grid search first
    coarse_best_rot = 0.0
    coarse_best_side = float('inf')
    n_coarse = 60
    for i in range(n_coarse):
        rot = i * 30.0 / n_coarse
        side = _min_side_for_rotation(rot)
        if side < coarse_best_side:
            coarse_best_side = side
            coarse_best_rot = rot
    
    # Fine golden-section search around the coarse best
    # Search in a window around the coarse best
    window = 30.0 / n_coarse
    a = max(0.0, coarse_best_rot - window)
    b = min(30.0, coarse_best_rot + window)
    
    if time.time() < budget - 20:
        fine_rot = _golden_section_search(_min_side_for_rotation, a, b, tol=1e-12, max_iter=80)
        fine_side = _min_side_for_rotation(fine_rot)
        
        if fine_side < coarse_best_side:
            coarse_best_rot = fine_rot
            coarse_best_side = fine_side
    
    # Also search the full [0, 30) range with multiple golden section searches
    if time.time() < budget - 10:
        # Try several windows to find global minimum
        for start in [0, 5, 10, 15, 20, 25]:
            if time.time() > budget - 5:
                break
            wa = max(0.0, float(start))
            wb = min(30.0, float(start) + 10.0)
            if wb > wa:
                r = _golden_section_search(_min_side_for_rotation, wa, wb, tol=1e-12, max_iter=50)
                s = _min_side_for_rotation(r)
                if s < coarse_best_side:
                    coarse_best_side = s
                    coarse_best_rot = r
    
    # Build the result with the optimal rotation
    seed_inner, seed_oc, seed_s, seed_rot = _hexagonal_seed()
    
    # Apply the optimal rotation to all inner hexagons
    rot_rad = math.radians(coarse_best_rot)
    rotated_inner = []
    for (x, y, a) in seed_inner:
        # Rotate position
        new_x = x * math.cos(rot_rad) - y * math.sin(rot_rad)
        new_y = x * math.sin(rot_rad) + y * math.cos(rot_rad)
        # Rotate orientation
        new_a = a + coarse_best_rot
        rotated_inner.append([new_x, new_y, new_a])
    
    s_optimal = _min_outer_side(rotated_inner, seed_oc, coarse_best_rot)
    
    if _valid(rotated_inner, seed_oc, s_optimal, coarse_best_rot):
        best = (rotated_inner, seed_oc, s_optimal, coarse_best_rot)
        best_score = s_optimal
    else:
        # Fallback to unrotated
        seed_s_tight = _min_outer_side(seed_inner, seed_oc, seed_rot)
        if _valid(seed_inner, seed_oc, seed_s_tight, seed_rot):
            best = (seed_inner, seed_oc, seed_s_tight, seed_rot)
            best_score = seed_s_tight
    
    # Phase 2: Try to improve with local optimization if time remains
    if best is not None and time.time() < budget - 5:
        inner_b, oc_b, s_b, rot_b = best
        refined = _refine(inner_b, oc_b, rot_b, rng, budget)
        if refined:
            inner_r, oc_r, s_r, rot_r = refined
            if s_r < best_score:
                best_score = s_r
                best = (inner_r, oc_r, s_r, rot_r)
    
    if best is None:
        inner, oc, s, rot = _hexagonal_seed()
        s_tight = _min_outer_side(inner, oc, rot)
        return inner, (oc[0], oc[1]), s_tight, rot
    
    inner, oc, s, rot = best
    s_tight = _min_outer_side(inner, oc, rot)
    if s_tight < s and _valid(inner, oc, s_tight, rot):
        s = s_tight
    
    return inner, (oc[0], oc[1]), s, rot
