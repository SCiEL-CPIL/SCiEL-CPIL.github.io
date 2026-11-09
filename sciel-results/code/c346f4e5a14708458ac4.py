"""Pack 12 unit hexagons in the smallest enclosing regular hexagon."""
import math
import time

N = 12
T0 = time.time()
DEADLINE = 112.0  # safety margin before 120 s wall-clock kill

# Unit hexagon: flat-top orientation, vertices at angles 0,60,...,300 deg.
# This gives 6 outward face normals: 0,60,120,180,240,300 deg.
_NORMS = [(math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))) for i in range(6)]
_VERTS = [(math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))) for i in range(6)]


def _hex_verts(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + ca * vx - sa * vy, cy + sa * vx + ca * vy) for (vx, vy) in _VERTS]


def _hex_normals(ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(ca * nx - sa * ny, sa * nx + ca * ny) for (nx, ny) in _NORMS]


def _enclosing_hexagon(points):
    """Return (cx, cy, S, ang_deg) of the minimal enclosing regular hexagon."""
    best = None
    for ang in range(0, 60, 5):
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        # Project points onto 6 directions and 6 opposite directions.
        dirs = []
        for (nx, ny) in _NORMS:
            dx, dy = ca * nx - sa * ny, sa * nx + ca * ny
            dirs.append((dx, dy))
        m_plus = [max(px * dx + py * dy for (px, py) in points) for (dx, dy) in dirs]
        m_minus = [min(px * dx + py * dy for (px, py) in points) for (dx, dy) in dirs]
        # Widths along each of the 6 axes.
        widths = [(m_plus[i] - m_minus[i]) / 2.0 for i in range(6)]
        S = max(widths)
        # Center: for each axis, center coordinate = (m_plus + m_minus)/2 along that axis.
        # Solve for (cx, cy) that satisfies all 6. Use least squares.
        A = []
        b = []
        for i in range(6):
            (dx, dy) = dirs[i]
            A.append([dx, dy])
            b.append((m_plus[i] + m_minus[i]) / 2.0)
        # Normal equations.
        Sxx = sum(r[0] * r[0] for r in A)
        Sxy = sum(r[0] * r[1] for r in A)
        Syy = sum(r[1] * r[1] for r in A)
        Sxb = sum(r[0] * v for r, v in zip(A, b))
        Syb = sum(r[1] * v for r, v in zip(A, b))
        det = Sxx * Syy - Sxy * Sxy
        if abs(det) < 1e-12:
            continue
        cx = (Syy * Sxb - Sxy * Syb) / det
        cy = (Sxx * Syb - Sxy * Sxb) / det
        if best is None or S < best[0]:
            best = (S, cx, cy, ang)
    return best


def _separation(v1, v2, ang1, ang2):
    """Squared minimum distance between two unit hexagons (0 if overlapping)."""
    p1 = _hex_verts(*v1[:2], v1[2])
    p2 = _hex_verts(*v2[:2], v2[2])
    # Edge-edge and vertex-edge distances. For convex polygons, min distance
    # is between edges. Use a simple check: if any vertex of one is inside the
    # other, or any edges cross, distance is 0. Otherwise min edge distance.
    # Simpler: sample. We'll do a robust convex-polygon distance via edges.
    def _dist_point_seg(px, py, ax, ay, bx, by):
        abx, aby = bx - ax, by - ay
        t = ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby + 1e-18)
        t = max(0.0, min(1.0, t))
        qx, qy = ax + t * abx, ay + t * aby
        return (px - qx) ** 2 + (py - qy) ** 2

    def _poly_edges(poly):
        m = len(poly)
        return [(poly[i], poly[(i + 1) % m]) for i in range(m)]

    e1 = _poly_edges(p1)
    e2 = _poly_edges(p2)
    best = math.inf
    for (a, b) in e1:
        for (c, d) in e2:
            # Segment-segment distance.
            # Check if segments intersect.
            def _ccw(o, p, q):
                return (q[1] - o[1]) * (p[0] - o[0]) - (p[1] - o[1]) * (q[0] - o[0])
            d1 = _ccw(c, d, a)
            d2 = _ccw(c, d, b)
            d3 = _ccw(a, b, c)
            d4 = _ccw(a, b, d)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return 0.0
            best = min(best, _dist_point_seg(*a, *c, *d))
            best = min(best, _dist_point_seg(*b, *c, *d))
            best = min(best, _dist_point_seg(*c, *a, *b))
            best = min(best, _dist_point_seg(*d, *a, *b))
    return best


def _objective(params):
    """params: 3*N floats. Returns (enclosing_side, min_separation)."""
    hexes = [(params[3 * i], params[3 * i + 1], params[3 * i + 2]) for i in range(N)]
    pts = []
    for (cx, cy, ang) in hexes:
        pts.extend(_hex_verts(cx, cy, ang))
    S, cx, cy, ang = _enclosing_hexagon(pts)
    min_sep = math.inf
    for i in range(N):
        for j in range(i + 1, N):
            s = _separation(hexes[i], hexes[j])
            if s < min_sep:
                min_sep = s
    return S, min_sep


def _local_optimize(params0, max_iter=400, tol=1e-9):
    """Simple projected gradient / subgradient minimization of S subject to min_sep >= 0."""
    import random
    params = list(params0)
    best = (None, None)
    S0, sep0 = _objective(params)
    if sep0 >= -1e-7:
        best = (S0, list(params))
    else:
        best = (S0, None)
    lr = 0.05
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        # Subgradient of S w.r.t. each vertex is 0.5 * direction of the max axis.
        # Approximate: use finite differences on a few coordinates.
        S, sep = _objective(params)
        if sep >= -1e-7 and (best[0] is None or S < best[0]):
            best = (S, list(params))
        # Compute gradient by finite difference on each coordinate.
        g = [0.0] * len(params)
        h = 1e-4
        for k in range(len(params)):
            sp = params[k]
            params[k] = sp + h
            Sp, _ = _objective(params)
            params[k] = sp - h
            Sm, _ = _objective(params)
            params[k] = sp
            g[k] = (Sp - Sm) / (2 * h)
        # Normalize gradient.
        gnorm = math.sqrt(sum(x * x for x in g))
        if gnorm < 1e-12:
            break
        # Step: decrease S, and if sep < 0 push apart.
        if sep < -1e-7:
            # Penalty: add gradient of -sep (push apart). Approx by random perturbation.
            # Instead, do a small random kick to break overlap.
            for k in range(len(params)):
                params[k] += random.uniform(-0.05, 0.05)
            continue
        for k in range(len(params)):
            params[k] -= lr * g[k] / gnorm
        if abs(params[k]) < 1e-12:
            pass
    return best


def _initial_config(seed=0):
    import random
    rng = random.Random(seed)
    # Place hexagons in a rough hexagonal lattice, then randomize.
    # Hexagonal packing of 12: use a 3-4-3-2 arrangement or a flower + ring.
    # Simple: place on a grid with spacing ~1.75 (slightly more than hex width 1.732).
    positions = []
    # 4 columns of 3, or a hexagonal cluster.
    # Use a compact cluster: center + 6 around + 5 more.
    base = [(0, 0)]
    for i in range(6):
        a = math.radians(60 * i)
        base.append((1.74 * math.cos(a), 1.74 * math.sin(a)))
    # Add 5 more on the next ring.
    for i in range(5):
        a = math.radians(60 * i + 30)
        base.append((3.4 * math.cos(a), 3.4 * math.sin(a)))
    params = []
    for (x, y) in base:
        params.append(x + rng.uniform(-0.1, 0.1))
        params.append(y + rng.uniform(-0.1, 0.1))
        params.append(rng.uniform(0, 60))
    return params


def construct():
    import random
    rng = random.Random(42)
    best_S = math.inf
    best_params = None
    # Multi-start local optimization.
    n_starts = 6
    for s in range(n_starts):
        if time.time() > DEADLINE - 5.0:
            break
        p0 = _initial_config(seed=s * 13 + 7)
        S, params = _local_optimize(p0, max_iter=200)
        if params is not None and S < best_S:
            best_S = S
            best_params = params
        # Also try a couple of random perturbations from the best.
        if best_params is not None:
            for _ in range(2):
                if time.time() > DEADLINE - 5.0:
                    break
                p = list(best_params)
                for k in range(len(p)):
                    p[k] += rng.uniform(-0.3, 0.3)
                S2, p2 = _local_optimize(p, max_iter=100)
                if p2 is not None and S2 < best_S:
                    best_S = S2
                    best_params = p2
    if best_params is None:
        # Fallback: the initial cluster.
        best_params = _initial_config(seed=0)
        best_S, _ = _objective(best_params)
    # Compute the enclosing hexagon.
    hexes = [(best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]) for i in range(N)]
    pts = []
    for (cx, cy, ang) in hexes:
        pts.extend(_hex_verts(cx, cy, ang))
    S, cx, cy, ang = _enclosing_hexagon(pts)
    inner = [[best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]] for i in range(N)]
    return inner, (cx, cy), S, ang
