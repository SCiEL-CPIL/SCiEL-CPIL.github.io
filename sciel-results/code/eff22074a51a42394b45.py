import math
import time
import numpy as np

N = 12
T0 = time.time()
DEADLINE = 112.0          # safety margin below 120 s
EPS = 1e-9

# ---- unit hexagon geometry (side 1, flat-top orientation at angle 0) ----
# vertices at angles 0,60,...,300 degrees, radius = 1
_HX = np.array([math.cos(math.radians(a)) for a in range(0, 360, 60)])
_HY = np.array([math.sin(math.radians(a)) for a in range(0, 360, 60)])
# supporting-function directions (6 edge normals) for a flat-top hexagon
_NX = np.array([math.cos(math.radians(a)) for a in range(30, 390, 60)])
_NY = np.array([math.sin(math.radians(a)) for a in range(30, 390, 60)])
# apothem (inradius) of unit hexagon = sqrt(3)/2
APO = math.sqrt(3.0) / 2.0

# ---- fixed non-overlapping honeycomb cluster of 12 unit hexagons ----
# axial coordinates (q, r) -> pixel (x, y) for pointy/flat honeycomb
# use flat-top honeycomb spacing: dx = 1.5, dy = sqrt(3)
_DX = 1.5
_DY = math.sqrt(3.0)
_AXIAL = [
    (0, 0),
    (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1),
    (0, -1), (1, -1), (2, -1),
    (1, 2), (2, 2),
    (1, -2), (2, -2),
]
# verify count
assert len(_AXIAL) == 12

def _axial_to_xy(q, r):
    return (q + 0.5 * r) * _DX, r * _DY

_cluster = np.array([_axial_to_xy(q, r) for (q, r) in _AXIAL])
# center the cluster at origin (subtract centroid) for symmetry
_cx = _cluster[:, 0].mean()
_cy = _cluster[:, 1].mean()
_cluster = _cluster - np.array([_cx, _cy])


def _hex_vertices(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    xs = s * _HX
    ys = s * _HY
    rx = ca * xs - sa * ys
    ry = sa * xs + ca * ys
    return np.column_stack([rx + cx, ry + cy])


def _support(points, dirx, diry):
    return float(np.max(points[:, 0] * dirx + points[:, 1] * diry))


def _cluster_support(theta):
    # cluster points rotated by theta, supporting function in 6 fixed directions
    a = math.radians(theta)
    ca, sa = math.cos(a), math.sin(a)
    rx = ca * _cluster[:, 0] - sa * _cluster[:, 1]
    ry = sa * _cluster[:, 0] + ca * _cluster[:, 1]
    proj = rx[:, None] * _NX[None, :] + ry[:, None] * _NY[None, :]
    return np.max(proj, axis=0)


def _side_length(theta):
    # minimal enclosing hexagon side length with FIXED flat-top axes
    h = _cluster_support(theta)          # cluster support per direction
    total = h + APO                      # Minkowski sum with unit hexagon
    return float(np.max(total) / APO)


def _check_valid(theta, s, ox, oy):
    # verify each inner hexagon is fully inside the outer hexagon
    outer = _hex_vertices(ox, oy, s, 0.0)
    for (qx, qy) in _cluster:
        # rotate hexagon by theta
        a = math.radians(theta)
        ca, sa = math.cos(a), math.sin(a)
        xs = _HX
        ys = _HY
        rx = ca * xs - sa * ys
        ry = sa * xs + ca * ys
        verts = np.column_stack([rx + qx, ry + qy])
        # point-in-hexagon test via half-planes
        for i in range(6):
            p0 = outer[i]
            p1 = outer[(i + 1) % 6]
            ex = p1[0] - p0[0]
            ey = p1[1] - p0[1]
            # inward normal
            nx = -ey
            ny = ex
            nrm = math.hypot(nx, ny)
            nx /= nrm
            ny /= nrm
            # distance from p0 to each vertex must be <= 0 (inside)
            d = (verts[:, 0] - p0[0]) * nx + (verts[:, 1] - p0[1]) * ny
            if np.max(d) > 1e-6:
                return False
    return True


def _build(theta):
    s = _side_length(theta)
    ox = 0.0
    oy = 0.0
    inner = []
    a = math.radians(theta)
    ca, sa = math.cos(a), math.sin(a)
    for (qx, qy) in _cluster:
        inner.append([qx, qy, theta])
    return inner, (ox, oy), s, 0.0


def _optimize_1d():
    # global search over theta in [0, 60)
    best_theta = None
    best_s = float('inf')
    # coarse sweep
    n_coarse = 240
    for i in range(n_coarse):
        th = (i + 0.5) * (60.0 / n_coarse)
        s = _side_length(th)
        if s < best_s:
            best_s = s
            best_theta = th
    # golden-section refinement
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    lo = max(0.0, best_theta - 60.0 / n_coarse)
    hi = min(60.0, best_theta + 60.0 / n_coarse)
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = _side_length(c)
    fd = _side_length(d)
    for _ in range(60):
        if time.time() > DEADLINE:
            break
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = _side_length(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = _side_length(d)
    theta = (lo + hi) / 2.0
    s = _side_length(theta)
    if s < best_s:
        best_s = s
        best_theta = theta
    return best_theta, best_s


def construct():
    # 1-D global search over orientation
    theta, s = _optimize_1d()
    inner, center, side, angle = _build(theta)

    # final high-precision polish + validity check
    if _check_valid(theta, side, center[0], center[1]):
        return inner, center, side, angle

    # fallback: expand side slightly until valid
    for k in range(200):
        side2 = side * (1.0 + 1e-4 * (k + 1))
        if _check_valid(theta, side2, center[0], center[1]):
            return inner, center, side2, angle
        if time.time() > DEADLINE:
            break

    # last resort: very large outer hexagon (valid but poor)
    return inner, (0.0, 0.0), 100.0, 0.0
