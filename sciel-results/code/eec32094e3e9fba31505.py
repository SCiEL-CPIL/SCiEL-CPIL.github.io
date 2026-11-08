import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 115.0  # safety margin before 150s kill

# 6 outward unit normals of a regular hexagon of side 1 (centered, axis-aligned)
# A regular hexagon of side s has apothem a = s*sqrt(3)/2
# Outward normals point to midpoints of edges, at angles 0, 60, 120, 180, 240, 300
def _hex_normals(theta_deg=0.0):
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    angles = np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]) + theta
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)  # shape (6,2)

def _hex_vertices(cx, cy, side, theta_deg=0.0):
    """Vertices of a regular hexagon (flat-top when theta=0)."""
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    # Regular hexagon vertices at angles 0,60,...,300 with radius = side
    angles = np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]) + theta
    R = side
    return np.stack([cx + R*np.cos(angles), cy + R*np.sin(angles)], axis=1)

def _inner_hex_verts(x, y, angle_deg):
    """Vertices of a unit regular hexagon centered at (x,y) rotated by angle_deg."""
    return _hex_vertices(x, y, 1.0, angle_deg)

def _outer_side_for_config(centers, angles, outer_theta_deg, cx, cy):
    """Minimal side length of outer hexagon (center (cx,cy), rotation outer_theta_deg)
    that contains all inner unit hexagons. Uses support function / half-space test."""
    normals = _hex_normals(outer_theta_deg)  # (6,2)
    # For each inner hex, get its 6 vertices
    max_proj = 0.0
    for (x, y, a) in zip(centers, angles):
        verts = _inner_hex_verts(x, y, a)  # (6,2)
        rel = verts - np.array([cx, cy])
        # projection onto each normal
        projs = rel @ normals.T  # (6,6)
        m = projs.max()
        if m > max_proj:
            max_proj = m
    # need s * sqrt(3)/2 >= max_proj  =>  s >= max_proj / (sqrt(3)/2)
    return max_proj / (math.sqrt(3.0) / 2.0)

def _overlap_penalty(centers, angles, eps=0.05):
    """Penalize pairwise overlap of inner hexagons.
    Two convex polygons overlap iff one vertex is inside the other OR edges intersect.
    For regular hexagons of same size, we check vertex-in-polygon both ways and edge crossings.
    """
    n = len(centers)
    verts = [_inner_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(n)]
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            vi, vj = verts[i], verts[j]
            # quick bbox reject
            if (vi[:,0].max() + eps < vj[:,0].min()) or (vj[:,0].max() + eps < vi[:,0].min()) or \
               (vi[:,1].max() + eps < vj[:,1].min()) or (vj[:,1].max() + eps < vi[:,1].min()):
                continue
            # vertex-in-polygon test (both ways)
            if _any_vertex_in_poly(vi, vj) or _any_vertex_in_poly(vj, vi):
                pen += 100.0
            else:
                # edge intersection
                if _edges_intersect(vi, vj):
                    pen += 100.0
    return pen

def _point_in_convex(pt, poly):
    """Check if point is inside (or on boundary) a convex polygon given in CCW order."""
    n = len(poly)
    sign = None
    for k in range(n):
        a = poly[k]
        b = poly[(k+1) % n]
        cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True

def _any_vertex_in_poly(vi, vj):
    for p in vi:
        if _point_in_convex(p, vj):
            return True
    return False

def _seg_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _edges_intersect(vi, vj):
    ni, nj = len(vi), len(vj)
    for k in range(ni):
        a, b = vi[k], vi[(k+1)%ni]
        for l in range(nj):
            c, d = vj[l], vj[(l+1)%nj]
            if _seg_intersect(a, b, c, d):
                return True
    return False

def _objective(params, outer_theta_deg):
    """params: [x0,y0,a0, x1,y1,a1, ...] for N hexagons."""
    if time.time() > DEADLINE:
        return 1e9
    centers = [(params[3*i], params[3*i+1]) for i in range(N)]
    angles = [params[3*i+2] for i in range(N)]
    side = _outer_side_for_config(centers, angles, outer_theta_deg, 0.0, 0.0)
    pen = _overlap_penalty(centers, angles)
    return side + pen

def _heuristic_lattice():
    """Hexagonal lattice of 12 unit hexagons: 4-5-3 or similar.
    Use a compact arrangement: rows of 5,4,3? Let's do a hexagonal cluster.
    A hexagonal number 12 isn't standard; use 4+5+3 = 12 or 5+4+3.
    Use rows: 3, 5, 4? Let's do a near-hexagonal: centers on triangular lattice spacing 2 (touching).
    Triangular lattice: spacing 2 in x, sqrt(3) in y for adjacent rows.
    """
    # Build a compact blob: rows of 4, 5, 3 = 12
    rows = [4, 5, 3]
    centers = []
    angles = [0.0]*N
    y = 0.0
    for r, count in enumerate(rows):
        for k in range(count):
            x = (k - (count-1)/2.0) * 2.0
            centers.append((x, y))
        y += math.sqrt(3)
    # center the blob
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    cx = (min(xs)+max(xs))/2
    cy = (min(ys)+max(ys))/2
    centers = [(x-cx, y-cy) for (x,y) in centers]
    return centers, angles

def _heuristic_ring():
    """1 center + 6 around + 5 more in a second partial ring."""
    centers = [(0.0, 0.0)]
    angles = [0.0]
    # 6 neighbors at distance 2
    for k in range(6):
        a = math.radians(60*k)
        centers.append((2.0*math.cos(a), 2.0*math.sin(a)))
        angles.append(0.0)
    # 5 more at distance ~4 in a ring
    for k in range(5):
        a = math.radians(60*k + 30)
        centers.append((4.0*math.cos(a), 4.0*math.sin(a)))
        angles.append(0.0)
    return centers, angles

def _heuristic_grid():
    """3x4 grid of touching hexagons (rotated 30 deg so they tile)."""
    # Flat-top hexagons tile with horizontal spacing 1.5*side and vertical sqrt(3)*side, offset.
    # For unit side, flat-top: x spacing = 1.5, y spacing = sqrt(3), odd rows offset 0.75
    centers = []
    angles = [0.0]*N
    cols, rows = 4, 3
    for r in range(rows):
        for c in range(cols):
            x = c*1.5 + (0.75 if r % 2 else 0.0)
            y = r*math.sqrt(3)
            centers.append((x, y))
    xs = [c[0] for c in centers]; ys = [c[1] for c in centers]
    cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2
    centers = [(x-cx, y-cy) for (x,y) in centers]
    return centers, angles

def _pack_from(centers, angles, outer_theta_deg, maxiter=400):
    """Optimize starting from given config using L-BFGS-B."""
    x0 = []
    for (x, y), a in zip(centers, angles):
        x0 += [x, y, a]
    res = minimize(_objective, x0, args=(outer_theta_deg,), method='L-BFGS-B',
                   options={'maxiter': maxiter, 'maxfun': maxiter*3})
    if time.time() > DEADLINE:
        return None
    return res

def construct():
    t0 = time.time()
    best = None
    best_score = float('inf')

    heuristics = [_heuristic_lattice, _heuristic_ring, _heuristic_grid]
    heur_results = []
    for h in heuristics:
        if time.time() > DEADLINE:
            break
        centers, angles = h()
        # quick score at outer_theta=0
        s = _outer_side_for_config(centers, angles, 0.0, 0.0, 0.0)
        pen = _overlap_penalty(centers, angles)
        score = s + pen
        if pen == 0.0:
            heur_results.append((score, centers, angles))
        if score < best_score:
            best_score = score
            best = (centers, angles, 0.0)

    # Optimize each feasible heuristic
    for (score, centers, angles) in sorted(heur_results):
        if time.time() > DEADLINE:
            break
        res = _pack_from(centers, angles, 0.0, maxiter=300)
        if res is None:
            continue
        x = res.x
        centers2 = [(x[3*i], x[3*i+1]) for i in range(N)]
        angles2 = [x[3*i+2] for i in range(N)]
        s = _outer_side_for_config(centers2, angles2, 0.0, 0.0, 0.0)
        pen = _overlap_penalty(centers2, angles2)
        if pen < 1e-6 and s < best_score:
            best_score = s
            best = (centers2, angles2, 0.0)

    # A few random restarts if time permits
    rng = np.random.default_rng(0)
    attempts = 0
    while time.time() < DEADLINE - 5.0 and attempts < 12:
        attempts += 1
        # random compact blob
        centers = []
        angles = []
        for i in range(N):
            r = rng.uniform(0, 4.0)
            a = rng.uniform(0, 2*math.pi)
            centers.append((r*math.cos(a), r*math.sin(a)))
            angles.append(rng.uniform(-30, 30))
        # center it
        xs = [c[0] for c in centers]; ys = [c[1] for c in centers]
        cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2
        centers = [(x-cx, y-cy) for (x,y) in centers]
        res = _pack_from(centers, angles, 0.0, maxiter=200)
        if res is None:
            continue
        x = res.x
        centers2 = [(x[3*i], x[3*i+1]) for i in range(N)]
        angles2 = [x[3*i+2] for i in range(N)]
        s = _outer_side_for_config(centers2, angles2, 0.0, 0.0, 0.0)
        pen = _overlap_penalty(centers2, angles2)
        if pen < 1e-6 and s < best_score:
            best_score = s
            best = (centers2, angles2, 0.0)

    if best is None:
        # fallback: return a valid (if large) configuration
        centers, angles = _heuristic_grid()
        s = _outer_side_for_config(centers, angles, 0.0, 0.0, 0.0)
        inner = [[centers[i][0], centers[i][1], math.degrees(angles[i])] for i in range(N)]
        return inner, (0.0, 0.0), s, 0.0

    centers, angles, outer_theta = best
    s = _outer_side_for_config(centers, angles, outer_theta, 0.0, 0.0)
    inner = [[centers[i][0], centers[i][1], math.degrees(angles[i])] for i in range(N)]
    return inner, (0.0, 0.0), s, outer_theta

