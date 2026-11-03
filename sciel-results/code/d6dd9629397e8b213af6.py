"""Unit regular hexagons inside a regular hexagon — Nelder-Mead with LP outer."""
import math
import time
import random

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin within 150 s limit


def _hex_vertices(cx, cy, angle_deg):
    ar = math.radians(angle_deg)
    return [(cx + math.cos(ar + math.pi * k / 3.0),
             cy + math.sin(ar + math.pi * k / 3.0)) for k in range(6)]


def _min_dist_poly_fast(poly1, poly2):
    """Minimum distance between two convex hexagons (0 if overlapping)."""
    # Bounding box quick reject
    minx1 = min(p[0] for p in poly1); maxx1 = max(p[0] for p in poly1)
    miny1 = min(p[1] for p in poly1); maxy1 = max(p[1] for p in poly1)
    minx2 = min(p[0] for p in poly2); maxx2 = max(p[0] for p in poly2)
    miny2 = min(p[1] for p in poly2); maxy2 = max(p[1] for p in poly2)

    if maxx1 < minx2 - 1e-9 or maxx2 < minx1 - 1e-9 or maxy1 < miny2 - 1e-9 or maxy2 < miny1 - 1e-9:
        # Disjoint bboxes: min vertex-to-vertex
        best = float('inf')
        for a in poly1:
            for b in poly2:
                d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if d < best:
                    best = d
        return math.sqrt(best)

    # Check overlap via vertex-in-polygon
    for v in poly1:
        if _pt_in_convex(v, poly2):
            return 0.0
    for v in poly2:
        if _pt_in_convex(v, poly1):
            return 0.0

    # Edge intersection check
    for i in range(6):
        for j in range(6):
            if _seg_intersect(poly1[i], poly1[(i+1) % 6], poly2[j], poly2[(j+1) % 6]):
                return 0.0

    # Min distance: vertex-to-vertex + vertex-to-edge
    best = float('inf')
    for a in poly1:
        for b in poly2:
            d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d < best:
                best = d
        for i in range(6):
            d = _pt_seg_dist2(a, poly2[i], poly2[(i+1) % 6])
            if d < best:
                best = d
    for b in poly2:
        for i in range(6):
            d = _pt_seg_dist2(b, poly1[i], poly1[(i+1) % 6])
            if d < best:
                best = d
    return math.sqrt(best)


def _pt_in_convex(pt, poly, eps=1e-9):
    px, py = pt
    for i in range(6):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % 6]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > eps:
            return False
    return True


def _seg_intersect(p1, p2, p3, p4, eps=1e-9):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    if abs(d1) <= eps and _on_seg(p3, p1, p4, eps): return True
    if abs(d2) <= eps and _on_seg(p3, p2, p4, eps): return True
    if abs(d3) <= eps and _on_seg(p1, p3, p2, eps): return True
    if abs(d4) <= eps and _on_seg(p1, p4, p2, eps): return True
    return False


def _on_seg(p, q, r, eps=1e-9):
    if (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and
        min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps and
        abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])) <= eps):
        return True
    return False


def _pt_seg_dist2(pt, a, b):
    ax, ay = a; bx, by = b; px, py = pt
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return (px - ax)**2 + (py - ay)**2
    t = ((px - ax)*dx + (py - ay)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return (px - cx)**2 + (py - cy)**2


def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """LP: minimum side length of outer hexagon with given angle containing all inner hex vertices."""
    from scipy.optimize import linprog

    all_verts = []
    for ix, iy, ia in inner_hexes:
        all_verts.extend(_hex_vertices(ix, iy, ia))

    c = [0.0, 0.0, 1.0]
    A_ub = []
    b_ub = []

    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        for vx, vy in all_verts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
        for vx, vy in all_verts:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
    if result.success:
        cx, cy, t = result.x
        return cx, cy, t / (math.sqrt(3) / 2)
    # fallback
    cx = sum(v[0] for v in all_verts) / len(all_verts)
    cy = sum(v[1] for v in all_verts) / len(all_verts)
    req = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        for vx, vy in all_verts:
            d = (vx - cx) * nx + (vy - cy) * ny
            req = max(req, abs(d))
    return cx, cy, req / (math.sqrt(3) / 2)


def _check_disjoint(inner_hexes, tol=1e-6):
    """Check all pairs are disjoint (non-overlapping)."""
    n = len(inner_hexes)
    polys = [_hex_vertices(h[0], h[1], h[2]) for h in inner_hexes]
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_dist_poly_fast(polys[i], polys[j])
            if d < tol:
                return False
    return True


def _objective(params, shared_angle=True):
    """Objective: outer hex side length + large penalty for overlap."""
    if time.time() > _DEADLINE - 3.0:
        return 9999.0

    n = N
    if shared_angle:
        inner = [[params[2*i], params[2*i+1], params[-1]] for i in range(n)]
    else:
        inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(n)]

    # Overlap check
    polys = [_hex_vertices(h[0], h[1], h[2]) for h in inner]
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_dist_poly_fast(polys[i], polys[j])
            if d < 0.001:
                penalty += (0.001 - d) ** 2

    if penalty > 1e-8:
        # Heavy penalty
        cx = sum(h[0] for h in inner) / n
        cy = sum(h[1] for h in inner) / n
        maxr = max(math.sqrt((h[0]-cx)**2 + (h[1]-cy)**2) + 1.0 for h in inner)
        outer = maxr / (math.sqrt(3)/2)
        return outer + 1000.0 * penalty

    # Compute best outer side over a few angles
    best_s = float('inf')
    for ang in [0.0, 15.0, 30.0, 45.0]:
        _, _, s = _min_enclosing_hex_side_lp(inner, ang)
        if s < best_s:
            best_s = s

    return best_s


def _nelder_mead(inner_hexes, shared_angle=True, max_iter=2000, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Nelder-Mead simplex method."""
    n = N
    nvars = 2 * n + 1 if shared_angle else 3 * n

    def to_params(inner):
        if shared_angle:
            p = []
            for h in inner:
                p.append(h[0]); p.append(h[1])
            p.append(inner[0][2])
            return p
        else:
            p = []
            for h in inner:
                p.extend(h)
            return p

    def from_params(p):
        if shared_angle:
            return [[p[2*i], p[2*i+1], p[-1]] for i in range(n)]
        else:
            return [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(n)]

    # Initial simplex
    x0 = to_params(inner_hexes)
    simplex = [x0[:]]
    fvals = [_objective(x0, shared_angle)]

    # Perturb each coordinate
    for i in range(nvars):
        xp = x0[:]
        step = 0.1 if i < 2 * n else 5.0  # larger step for angle
        xp[i] += step
        simplex.append(xp)
        fvals.append(_objective(xp, shared_angle))

    # Sort
    order = sorted(range(nvars + 1), key=lambda i: fvals[i])
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]

    for it in range(max_iter):
        if time.time() > _DEADLINE - 3.0:
            break

        # Reflect
        centroid = [sum(simplex[i][j] for i in range(nvars)) / nvars for j in range(nvars)]
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(nvars)]
        fr = _objective(xr, shared_angle)

        if fr < fvals[0]:
            # Expand
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(nvars)]
            fe = _objective(xe, shared_angle)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            # Contract
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(nvars)]
            fc = _objective(xc, shared_angle)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, nvars + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(nvars)]
                    fvals[i] = _objective(simplex[i], shared_angle)

        # Re-sort
        order = sorted(range(nvars + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

    best_params = simplex[0]
    best_inner = from_params(best_params)
    return best_inner


def construct():
    global _DEADLINE

    if time.time() > _DEADLINE - 2.0:
        inner = [[0.0, 0.0, 30.0]] * N
        return inner, (0.0, 0.0), 50.0, 0.0

    sqrt3 = math.sqrt(3)
    scale = 1.001

    # Build hexagonal lattice cluster: center + 6 neighbors + 4 from second ring
    # Axial coords: (i, j) -> (i*sqrt3 + j*sqrt3/2, j*1.5)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = (i * sqrt3 + j * sqrt3 / 2) * scale
            y = j * 1.5 * scale
            d2 = x * x + y * y
            lattice_points.append((d2, x, y))
    lattice_points.sort(key=lambda t: t[0])

    seen = set()
    positions = []
    for d2, x, y in lattice_points:
        key = (round(x, 6), round(y, 6))
        if key in seen:
            continue
        seen.add(key)
        positions.append((x, y))
        if len(positions) == N:
            break

    if len(positions) < N:
        positions = [(k * sqrt3 * scale, 0.0) for k in range(N)]

    # Initial: all at 30 degrees (pointy-top)
    initial_inner = [[px, py, 30.0] for px, py in positions]

    # Compute initial enclosing hex
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_inner = initial_inner

    for ang in [0.0, 15.0, 30.0, 45.0]:
        cx, cy, s = _min_enclosing_hex_side_lp(initial_inner, ang)
        if s < best_side:
            best_side = s
            best_center = (cx, cy)
            best_angle = ang

    # --- Nelder-Mead optimization ---
    # Phase 1: shared angle (23 vars)
    if time.time() < _DEADLINE - 30.0:
        polished = _nelder_mead(initial_inner, shared_angle=True, max_iter=1500)
        # Verify disjointness
        if _check_disjoint(polished):
            for ang in [0.0, 15.0, 30.0, 45.0]:
                cx, cy, s = _min_enclosing_hex_side_lp(polished, ang)
                if s < best_side:
                    best_side = s
                    best_center = (cx, cy)
                    best_angle = ang
                    best_inner = polished
        else:
            # Try to fix by scaling positions slightly
            for s in [1.005, 1.01, 1.02]:
                fixed = [[h[0]*s, h[1]*s, h[2]] for h in polished]
                if _check_disjoint(fixed):
                    for ang in [0.0, 15.0, 30.0, 45.0]:
                        cx, cy, side = _min_enclosing_hex_side_lp(fixed, ang)
                        if side < best_side:
                            best_side = side
                            best_center = (cx, cy)
                            best_angle = ang
                            best_inner = fixed
                    break

    # Phase 2: per-hex angle (33 vars) if time permits
    if time.time() < _DEADLINE - 25.0:
        polished2 = _nelder_mead(best_inner, shared_angle=False, max_iter=800)
        if _check_disjoint(polished2):
            for ang in [0.0, 15.0, 30.0, 45.0]:
                cx, cy, s = _min_enclosing_hex_side_lp(polished2, ang)
                if s < best_side:
                    best_side = s
                    best_center = (cx, cy)
                    best_angle = ang
                    best_inner = polished2

    # Phase 3: Multi-start with random perturbations
    if time.time() < _DEADLINE - 15.0:
        random.seed(123)
        for trial in range(4):
            if time.time() > _DEADLINE - 10.0:
                break
            perturbed = []
            for h in best_inner:
                px = h[0] + random.uniform(-0.08, 0.08)
                py = h[1] + random.uniform(-0.08, 0.08)
                pa = h[2] + random.uniform(-3.0, 3.0)
                perturbed.append([px, py, pa])
            p3 = _nelder_mead(perturbed, shared_angle=True, max_iter=500)
            if _check_disjoint(p3):
                for ang in [0.0, 15.0, 30.0, 45.0]:
                    cx, cy, s = _min_enclosing_hex_side_lp(p3, ang)
                    if s < best_side:
                        best_side = s
                        best_center = (cx, cy)
                        best_angle = ang
                        best_inner = p3

    # Phase 4: Fine angle sweep
    if time.time() < _DEADLINE - 5.0:
        for k in range(60):
            ang = (best_angle + 30.0 * (k / 59.0 - 0.5)) % 60
            cx, cy, s = _min_enclosing_hex_side_lp(best_inner, ang)
            if s < best_side:
                best_side = s
                best_center = (cx, cy)
                best_angle = ang

    # Even finer
    if time.time() < _DEADLINE - 3.0:
        for k in range(40):
            ang = (best_angle + 5.0 * (k / 39.0 - 0.5)) % 60
            cx, cy, s = _min_enclosing_hex_side_lp(best_inner, ang)
            if s < best_side:
                best_side = s
                best_center = (cx, cy)
                best_angle = ang

    return best_inner, (best_center[0], best_center[1]), best_side, best_angle
