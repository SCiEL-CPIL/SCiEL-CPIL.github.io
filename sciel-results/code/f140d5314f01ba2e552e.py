"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (pointy-top orientation at 0 deg)."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _outer_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _project_range(verts, axis):
    """Min and max projection of polygon vertices onto a unit axis."""
    dots = verts @ axis
    return dots.min(), dots.max()

def _sat_overlap(v1, v2, tol=1e-9):
    """True if two convex polygons overlap (interior intersection)."""
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            r1min, r1max = _project_range(v1, normal)
            r2min, r2max = _project_range(v2, normal)
            if r1max < r2min + tol or r2max < r1min + tol:
                return False
    return True

def _point_in_convex(px, py, poly, tol=1e-9):
    """Check if point is inside a convex polygon (given in CCW order)."""
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        ex = poly[j, 0] - poly[i, 0]
        ey = poly[j, 1] - poly[i, 1]
        px_rel = px - poly[i, 0]
        py_rel = py - poly[i, 1]
        cross = ex * py_rel - ey * px_rel
        if cross < -tol:
            return False
    return True

def _inner_fits_outer(inner_verts, outer_verts, tol=1e-9):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    for v in inner_verts:
        if not _point_in_convex(v[0], v[1], outer_verts, tol):
            return False
    return True

def _feasible(outer_side, placements, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Check if given placements fit inside outer hexagon and don't overlap each other."""
    ox, oy = outer_center
    outer_verts = _outer_vertices(ox, oy, outer_side, outer_angle)
    
    inner_verts_list = []
    for (ix, iy, iangle) in placements:
        inner_verts_list.append(_hex_vertices(ix, iy, iangle, 1.0))
    
    for iv in inner_verts_list:
        if not _inner_fits_outer(iv, outer_verts, tol=1e-9):
            return False
    
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(inner_verts_list[i], inner_verts_list[j], tol=1e-9):
                return False
    
    return True

def _min_poly_distance(v1, v2):
    """Minimum distance between two convex polygons. Returns 0 if overlapping."""
    if _sat_overlap(v1, v2, tol=1e-9):
        return 0.0
    min_dist = float('inf')
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1 = v1[i]
        for j in range(n2):
            p2 = v2[j]
            d = np.linalg.norm(p1 - p2)
            min_dist = min(min_dist, d)
            # p1 to edge v2[j]->v2[j+1]
            jn = (j + 1) % n2
            e0, e1 = v2[j], v2[jn]
            ev = e1 - e0
            el2 = np.dot(ev, ev)
            if el2 > 1e-12:
                t = np.dot(p1 - e0, ev) / el2
                t = max(0.0, min(1.0, t))
                proj = e0 + t * ev
                min_dist = min(min_dist, np.linalg.norm(p1 - proj))
            # p2 to edge v1[i]->v1[i+1]
            in_ = (i + 1) % n1
            e0, e1 = v1[i], v1[in_]
            ev = e1 - e0
            el2 = np.dot(ev, ev)
            if el2 > 1e-12:
                t = np.dot(p2 - e0, ev) / el2
                t = max(0.0, min(1.0, t))
                proj = e0 + t * ev
                min_dist = min(min_dist, np.linalg.norm(p2 - proj))
    return min_dist

def _penalty(x, outer_verts):
    """
    Continuous penalty for Nelder-Mead.
    x: array of length 3*N = [x0,y0,ang0, x1,y1,ang1, ...]
    Returns 0.0 if feasible, positive penalty otherwise.
    """
    penalty = 0.0
    verts_list = []
    for i in range(N):
        cx, cy, ang = x[3*i], x[3*i+1], x[3*i+2]
        v = _hex_vertices(cx, cy, ang, 1.0)
        verts_list.append(v)
        # Boundary: each vertex must be inside outer
        for pt in v:
            if not _point_in_convex(pt[0], pt[1], outer_verts, tol=1e-9):
                penalty += 5.0
    # Overlap: use min distance; penalize when distance < margin
    margin = 0.01
    for i in range(N):
        for j in range(i+1, N):
            d = _min_poly_distance(verts_list[i], verts_list[j])
            if d < margin:
                penalty += 10.0 * (margin - d)
    return penalty

def _make_initial_layout(outer_side, layout_type='hex_tiling', outer_center=(0.0, 0.0)):
    """
    Generate structured initial layouts for 12 unit hexagons.
    Returns array of shape (3*N,) with [x, y, angle_deg] per hexagon.
    """
    x = np.zeros(3 * N)
    ox, oy = outer_center
    # Inradius of outer hexagon
    r_in = outer_side * math.sqrt(3) / 2.0
    # Max center radius so hexagon fits
    max_r = max(0.05, r_in - 1.0)
    
    if layout_type == 'hex_tiling':
        # Hexagonal tiling: 1 center + 6 ring + 5 more arranged
        # For 12: use a compact hexagonal arrangement
        # Row-based hex packing: rows of 3,4,3,2 or similar
        # Let's do: center, then 6 around, then 5 more in a ring
        positions = []
        # Center
        positions.append((0, 0))
        # Ring 1: 6 hexagons at distance 2 (edge-to-edge for unit hexagons)
        d = 2.0  # center-to-center distance for touching unit hexagons
        for k in range(6):
            theta = k * math.pi / 3.0
            positions.append((d * math.cos(theta), d * math.sin(theta)))
        # Ring 2: 5 more hexagons
        d2 = 2.0 * math.sqrt(3)  # ~3.46
        for k in range(5):
            theta = k * 2 * math.pi / 5.0 + math.pi / 5.0
            positions.append((d2 * math.cos(theta), d2 * math.sin(theta)))
        
        # Scale to fit
        scale = 1.0
        for px, py in positions:
            r = math.sqrt(px*px + py*py)
            if r > max_r:
                scale = min(scale, max_r / r)
        
        for i, (px, py) in enumerate(positions[:N]):
            x[3*i] = ox + scale * px
            x[3*i+1] = oy + scale * py
            x[3*i+2] = 0.0  # all same orientation
    
    elif layout_type == 'grid':
        # 4x3 grid-like arrangement
        dx = 2.0  # horizontal spacing
        dy = math.sqrt(3)  # vertical spacing for hex grid
        cols = 4
        rows = 3
        count = 0
        for r in range(rows):
            for c in range(cols):
                if count >= N:
                    break
                px = (c - 1.5) * dx
                py = (r - 1.0) * dy
                # Offset odd rows
                if r % 2 == 1:
                    px += dx / 2.0
                x[3*count] = ox + px
                x[3*count+1] = oy + py
                x[3*count+2] = 0.0
                count += 1
    
    elif layout_type == 'ring':
        # All 12 in a ring around center
        for i in range(N):
            theta = i * 2 * math.pi / N
            x[3*i] = ox + max_r * 0.8 * math.cos(theta)
            x[3*i+1] = oy + max_r * 0.8 * math.sin(theta)
            x[3*i+2] = 0.0
    
    elif layout_type == 'random':
        for i in range(N):
            r = max_r * math.sqrt(np.random.random())
            theta = 2 * math.pi * np.random.random()
            x[3*i] = ox + r * math.cos(theta)
            x[3*i+1] = oy + r * math.sin(theta)
            x[3*i+2] = np.random.random() * 360.0
    
    return x

def _nm_optimize(x0, outer_verts, time_budget=5.0, maxiter=200):
    """
    Nelder-Mead local optimization to minimize penalty.
    Returns (best_x, best_penalty).
    """
    start = time.time()
    
    def objective(x):
        if time.time() - start > time_budget:
            return 1e10
        return _penalty(x, outer_verts)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8,
                                'adaptive': True})
        best_x = res.x
        best_p = _penalty(best_x, outer_verts)
    except Exception:
        best_x = x0
        best_p = _penalty(x0, outer_verts)
    
    return best_x, best_p

def _try_pack(outer_side, outer_center, outer_angle, time_budget=8.0, seed=42):
    """
    Try to pack N unit hexagons into outer hexagon.
    Returns (placements, success) where placements is list of (x, y, angle_deg).
    """
    np.random.seed(seed)
    outer_verts = _outer_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    best_x = None
    best_p = float('inf')
    
    layouts = ['hex_tiling', 'grid', 'ring', 'random', 'random']
    time_per = max(0.5, time_budget / len(layouts))
    
    for layout in layouts:
        if time.time() - time.time() > time_budget:
            break
        x0 = _make_initial_layout(outer_side, layout, outer_center)
        bx, bp = _nm_optimize(x0, outer_verts, time_budget=time_per)
        if bp < best_p:
            best_p = bp
            best_x = bx
        if bp < 1e-6:
            break
    
    if best_x is None:
        return None, False
    
    # Verify feasibility
    placements = []
    for i in range(N):
        placements.append((best_x[3*i], best_x[3*i+1], best_x[3*i+2] % 360.0))
    
    if _feasible(outer_side, placements, outer_center, outer_angle):
        return placements, True
    return None, False

def _bisection_search(lo, hi, outer_center, outer_angle, time_budget=100.0):
    """
    Bisection search for minimum outer side length.
    """
    start = time.time()
    best_placements = None
    best_side = hi
    
    # Verify hi is feasible
    found = False
    for seed in range(5):
        if time.time() - start > time_budget - 15:
            break
        placements, ok = _try_pack(hi, outer_center, outer_angle, time_budget=8.0, seed=seed)
        if ok:
            best_placements = placements
            best_side = hi
            found = True
            break
    
    if not found:
        hi *= 1.3
        for seed in range(5):
            if time.time() - start > time_budget - 15:
                break
            placements, ok = _try_pack(hi, outer_center, outer_angle, time_budget=8.0, seed=seed)
            if ok:
                best_placements = placements
                best_side = hi
                found = True
                break
    
    if not found:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Bisection iterations
    for it in range(12):
        if time.time() - start > time_budget - 10:
            break
        mid = (lo + hi) / 2.0
        found_mid = False
        for seed in range(3):
            if time.time() - start > time_budget - 5:
                break
            placements, ok = _try_pack(mid, outer_center, outer_angle, time_budget=5.0, seed=seed)
            if ok:
                found_mid = True
                best_placements = placements
                best_side = mid
                break
        
        if found_mid:
            hi = mid
        else:
            lo = mid
    
    return best_placements, (0.0, 0.0), best_side, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Area lower bound: S >= sqrt(6) ≈ 2.449
    # Practical lower bound with packing: ~3.5-4.0
    lo = 2.5
    hi = 10.0
    
    best_result = None
    best_side = float('inf')
    
    # Try outer angles: 0 and 30 degrees
    for outer_angle in [0.0, 30.0]:
        remaining = deadline - time.time()
        if remaining < 20.0:
            break
        inner, center, side, angle = _bisection_search(lo, hi, (0.0, 0.0), outer_angle,
                                                       time_budget=remaining - 5.0)
        if side < best_side:
            best_side = side
            best_result = (inner, center, side, angle)
    
    if best_result is not None:
        return best_result
    else:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
