"""Hexagon packing: rigid hexagonal cluster + global continuous optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

# Unit hexagon vertices (flat-top), side length 1
_HEX = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3) / 2],
    [-0.5, math.sqrt(3) / 2],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3) / 2],
    [0.5, -math.sqrt(3) / 2],
])
# Hexagon support directions (unit normals to the 6 sides), flat-top
_NORMS = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3) / 2],
    [-0.5, math.sqrt(3) / 2],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3) / 2],
    [0.5, -math.sqrt(3) / 2],
])


def _hex_vertices(cx, cy, ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return R @ _HEX.T + np.array([cx, cy])[:, None]


def _cluster_vertices(theta):
    """Build all inner hexagon vertices from parameter vector theta.
    theta layout: [x0,y0,a0, x1,y1,a1, ... , phi]"""
    n = N
    verts = []
    for i in range(n):
        x, y, a = theta[3 * i], theta[3 * i + 1], theta[3 * i + 2]
        verts.append(_hex_vertices(x, y, a))
    return np.concatenate(verts, axis=1)


def _pairwise_collision_penalty(theta, scale=30.0):
    """Penalty if any two inner hexagons overlap (edge intersection or containment)."""
    n = N
    polys = []
    for i in range(n):
        x, y, a = theta[3 * i], theta[3 * i + 1], theta[3 * i + 2]
        polys.append(_hex_vertices(x, y, a))
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pen += _poly_overlap_penalty(polys[i], polys[j], scale)
    return pen


def _seg_intersect(a, b, c, d):
    """Return True if segment (a,b) intersects segment (c,d)."""
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 1 if v > 1e-12 else (-1 if v < -1e-12 else 0)
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return False


def _point_in_poly(pt, poly):
    """Ray casting point-in-polygon test. poly is 2x6 array."""
    x, y = pt[0], pt[1]
    inside = False
    j = 5
    for i in range(6):
        xi, yi = poly[0, i], poly[1, i]
        xj, yj = poly[0, j], poly[1, j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _poly_overlap_penalty(p1, p2, scale):
    """Penalty for overlap between two convex hexagons p1, p2 (2x6)."""
    # Check edge-edge intersections
    for i in range(6):
        a, b = p1[:, i], p1[:, (i + 1) % 6]
        for j in range(6):
            c, d = p2[:, j], p2[:, (j + 1) % 6]
            if _seg_intersect(a, b, c, d):
                return scale
    # Check containment (center of p1 inside p2, or p2 inside p1)
    c1 = p1.mean(axis=1)
    c2 = p2.mean(axis=1)
    if _point_in_poly(c1, p2) or _point_in_poly(c2, p1):
        return scale
    return 0.0


def _minimal_container_side(theta):
    """Compute minimal regular hexagon side length containing all inner hexagons,
    with container angle phi = theta[-1]."""
    phi = theta[-1]
    verts = _cluster_vertices(theta)  # 2 x (6N)
    # Container normals: rotate _NORMS by phi
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, -s], [s, c]])
    cnorms = R @ _NORMS.T  # 2 x 6
    # For each normal n, width = max(d.n) - min(d.n)
    s_min = 0.0
    for k in range(6):
        n = cnorms[:, k]
        proj = verts[0, :] * n[0] + verts[1, :] * n[1]
        w = proj.max() - proj.min()
        # For a regular hexagon of side s with normal n (unit), width = 2s
        # because distance between opposite sides = 2s (apothem*2 = s*sqrt(3)... )
        # Actually width between two opposite sides along normal = 2 * apothem? 
        # Apothem = (sqrt(3)/2)*s. Distance between opposite sides = 2*apothem = sqrt(3)*s.
        # So width = sqrt(3)*s  =>  s = width / sqrt(3)
        s_min = max(s_min, w / math.sqrt(3))
    return s_min


def _objective(theta):
    side = _minimal_container_side(theta)
    pen = _pairwise_collision_penalty(theta)
    return side + pen


def _initial_theta():
    """Hexagonal cluster: center + ring of 6, plus 4 more in a second partial ring.
    Positions in hex lattice coords, converted to cartesian."""
    sq3 = math.sqrt(3)
    # Hex lattice basis: e1 = (1,0)*2? For unit hexagons touching, center distance = sqrt(3)
    # Nearest neighbor distance for flat-top hexagons = 2*apothem = sqrt(3)
    d = sq3  # center-to-center distance for touching
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # Ring 1: 6 neighbors
    for k in range(6):
        ang = math.pi / 3 * k
        positions.append((d * math.cos(ang), d * math.sin(ang)))
    # Ring 2: add 4 more (partial) at distance 2d in 4 directions
    for k in range(0, 6, 1):
        if len(positions) >= N:
            break
        ang = math.pi / 3 * k
        positions.append((2 * d * math.cos(ang), 2 * d * math.sin(ang)))
    positions = positions[:N]
    theta = []
    for (x, y) in positions:
        theta.extend([x, y, 0.0])
    theta.append(0.0)  # container angle phi
    return np.array(theta, dtype=float)


def _best_known():
    return None


def construct():
    global START, DEADLINE
    deadline = START + (DEADLINE - START)  # absolute
    deadline = 145.0

    best_theta = _initial_theta()
    best_val = _objective(best_theta)
    best_side = _minimal_container_side(best_theta)

    # Try multiple random initializations + Nelder-Mead
    rng = np.random.default_rng(42)
    ntrials = 6
    for t in range(ntrials):
        if time.time() > deadline - 20.0:
            break
        # Random perturbation of initial cluster
        th = _initial_theta().copy()
        th[:3 * N] += rng.uniform(-0.3, 0.3, size=3 * N)
        res = minimize(_objective, th, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-5})
        if time.time() > deadline - 15.0:
            break
        if res.fun < best_val:
            best_val = res.fun
            best_theta = res.x.copy()
            best_side = _minimal_container_side(best_theta)

    # Refine with tighter Nelder-Mead from best
    if time.time() < deadline - 10.0:
        res = minimize(_objective, best_theta, method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-5, 'fatol': 1e-6})
        if res.fun < best_val:
            best_val = res.fun
            best_theta = res.x.copy()
            best_side = _minimal_container_side(best_theta)

    # Build inner list from best_theta
    inner = []
    for i in range(N):
        x = best_theta[3 * i]
        y = best_theta[3 * i + 1]
        a = best_theta[3 * i + 2]
        inner.append([x, y, math.degrees(a)])

    side = _minimal_container_side(best_theta)
    phi = best_theta[-1]

    # Final safety: ensure no collisions; if collision, fall back to safe large side
    pen = _pairwise_collision_penalty(best_theta)
    if pen > 1.0:
        # Fall back to a guaranteed-valid spread-out configuration
        d = math.sqrt(3)
        inner_safe = []
        positions = []
        positions.append((0.0, 0.0))
        for k in range(6):
            ang = math.pi / 3 * k
            positions.append((d * math.cos(ang), d * math.sin(ang)))
        for k in range(6):
            if len(positions) >= N:
                break
            ang = math.pi / 3 * k
            positions.append((2 * d * math.cos(ang), 2 * d * math.sin(ang)))
        for (x, y) in positions[:N]:
            inner_safe.append([x, y, 0.0])
        # Safe side: bounding
        verts = []
        for (x, y, a) in inner_safe:
            verts.append(_hex_vertices(x, y, a))
        allv = np.concatenate(verts, axis=1)
        s_safe = 0.0
        for k in range(6):
            n = _NORMS[:, k]
            proj = allv[0, :] * n[0] + allv[1, :] * n[1]
            s_safe = max(s_safe, (proj.max() - proj.min()) / math.sqrt(3))
        return inner_safe, (0.0, 0.0), s_safe, 0.0

    return inner, (0.0, 0.0), side, math.degrees(phi)


if __name__ == "__main__":
    r = construct()
    print("side =", r[2])
