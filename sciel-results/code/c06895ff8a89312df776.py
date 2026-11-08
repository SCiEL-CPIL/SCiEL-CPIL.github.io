"""Discrete combinatorial packing of 12 unit hexagons in a minimal enclosing regular hexagon.

Approach:
  * Inner hexagons are restricted to vertices of a fine triangular lattice (spacing g)
    and to a small set of orientation angles (multiples of 6 degrees).
  * For any candidate configuration we compute the exact minimal enclosing regular
    hexagon side length (outer hexagon may translate and rotate) via convex hull +
    binary search exploiting the 60-degree symmetry of a regular hexagon.
  * A multi-start local search (simulated annealing) minimises that side length.
"""
import math
import random
import numpy as np

N = 12
G = 0.05                       # lattice spacing
L = 5.2                        # half-extent of the search box (squared area 20.7)
ANGLES = list(range(0, 60, 6)) # orientation candidates (degrees), 6-fold symmetry
TWO_PI = 2.0 * math.pi

# Pre-compute the 6 vertices of a unit regular hexagon for every allowed angle
_HEX_VERTS = np.zeros((len(ANGLES), 6, 2), dtype=float)
for _i, _deg in enumerate(ANGLES):
    _a = math.radians(_deg)
    for _k in range(6):
        _t = _a + _k * math.pi / 3.0
        _HEX_VERTS[_i, _k, 0] = math.cos(_t)
        _HEX_VERTS[_i, _k, 1] = math.sin(_t)


def _orient_index(angle_deg):
    return int(round(angle_deg)) % 60 // 6


def _points(pos, ang_idx):
    """Corner points of a unit hexagon at (pos, ang_idx)."""
    return pos[None, :, :] + _HEX_VERTS[ang_idx]


def _hull(pts):
    """Convex hull (Andrew's monotone chain) of a set of 2D points."""
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2:
        return np.array(pts, dtype=float)
    lower = []
    for p in pts:
        while len(lower) >= 2 and np.cross(np.array(lower[-1]) - np.array(lower[-2]),
                                           np.array(p) - np.array(lower[-1])) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and np.cross(np.array(upper[-1]) - np.array(upper[-2]),
                                           np.array(p) - np.array(upper[-1])) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _hex_width(theta):
    """Width of a unit regular hexagon (circumradius 1) in direction theta."""
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    return c + math.sqrt(3.0) * s


def _enclosing_side(pts, theta):
    """Minimal side length of a regular hexagon (outer, circumradius = side) with
    outward normal direction theta that encloses the given points."""
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-math.sin(theta), math.cos(theta)])
    d = pts @ u
    e = pts @ v
    lo, hi = float(d.min()), float(d.max())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        w = _hex_width(theta)
        if e.min() >= mid - w and e.max() <= mid + w:
            hi = mid
        else:
            lo = mid
    mid = 0.5 * (lo + hi)
    return (hi - lo) + 2.0 * math.sqrt(3.0) * abs(mid)


def _score(pos, ang):
    """Minimal enclosing regular hexagon side length for the configuration."""
    pts = np.vstack([_points(pos[i], ang[i]) for i in range(N)])
    hull = _hull(pts)
    best = math.inf
    for k in range(6):
        a = hull[k]
        b = hull[(k + 1) % len(hull)]
        d = b - a
        n = np.array([-d[1], d[0]])
        n = n / (np.linalg.norm(n) + 1e-12)
        if n @ a < 0:          # ensure outward normal
            n = -n
        best = min(best, _enclosing_side(hull, float(math.atan2(n[1], n[0]))))
    return best


def _init_state(rng):
    pos = np.array([[rng.uniform(-L, L), rng.uniform(-L, L)] for _ in range(N)])
    ang = np.array([rng.choice(len(ANGLES)) for _ in range(N)])
    return pos, ang


def _perturb(pos, ang, rng):
    i = rng.randrange(N)
    r = rng.random()
    if r < 0.7:
        pos[i, 0] = min(L, max(-L, pos[i, 0] + rng.gauss(0, 0.25)))
        pos[i, 1] = min(L, max(-L, pos[i, 1] + rng.gauss(0, 0.25)))
    else:
        ang[i] = (ang[i] + rng.choice([-1, 1])) % len(ANGLES)


def _local_search(pos, ang, rng, T0, T1, iters):
    cur = _score(pos, ang)
    best = cur
    for it in range(iters):
        T = T0 + (T1 - T0) * (it / float(iters))
        np2 = pos.copy()
        ag2 = ang.copy()
        _perturb(np2, ag2, rng)
        cand = _score(np2, ag2)
        if cand < cur or rng.random() < math.exp((cur - cand) / T):
            pos, ang, cur = np2, ag2, cand
            if cand < best:
                best = cand
    return pos, ang, best


def construct():
    rng = random.Random(12345)
    best_val = math.inf
    best_pos = None
    best_ang = None
    for s in range(12):
        p, a = _init_state(rng)
        p, a, val = _local_search(p, a, rng, 1.0, 0.02, 450)
        if val < best_val:
            best_val, best_pos, best_ang = val, p.copy(), a.copy()
    inner = [[float(best_pos[i, 0]), float(best_pos[i, 1]),
              float(ANGLES[int(best_ang[i])])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_val), 0.0
