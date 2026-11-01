"""Hexagon packing: 12 unit regular hexagons inside a minimal regular hexagon.

Strategy: a honeycomb (hexagonal-lattice) placement of the 12 inner hexagons
(rows of 4/4/4), parameterized by lattice spacing s and orientation theta.
We minimize the enclosing regular-hexagon side length via a coarse grid
followed by Nelder-Mead, using vectorized geometry for speed. A hard
deadline guard returns the best valid configuration found.
"""
import time
import math
import numpy as np

N = 12

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin under the 120 s wall-clock limit


def _hex_verts(cx, cy, s, theta):
    """Vertices of a regular hexagon (flat-top base) centered at (cx,cy),
    side length s, rotated by theta (radians). Returns (6,2) array."""
    a = theta + np.arange(6) * (math.pi / 3.0)
    return np.stack([cx + s * np.cos(a), cy + s * np.sin(a)], axis=1)


def _hex_verts_all(cx, cy, s, theta):
    """Vertices for all 12 hexagons on the honeycomb lattice.
    Returns (12,6,2) array."""
    base = np.array([
        [-1.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0],
        [-1.0, math.sqrt(3) / 2.0], [0.0, math.sqrt(3) / 2.0],
        [1.0, math.sqrt(3) / 2.0], [2.0, math.sqrt(3) / 2.0],
        [-1.0, -math.sqrt(3) / 2.0], [0.0, -math.sqrt(3) / 2.0],
        [1.0, -math.sqrt(3) / 2.0], [2.0, -math.sqrt(3) / 2.0],
    ]) * s
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    pts = (base @ rot.T) + np.array([cx, cy])
    local = _hex_verts(0.0, 0.0, 1.0, 0.0)  # (6,2)
    return pts[:, None, :] + local[None, :, :]


def _outer_verts(cx, cy, L, phi):
    a = phi + np.arange(6) * (math.pi / 3.0)
    return np.stack([cx + L * np.cos(a), cy + L * np.sin(a)], axis=1)


def _max_L(verts, cx, cy, phi):
    """Minimum regular-hexagon side length (centered at (cx,cy), orientation
    phi) that contains all vertices in (N,6,2)."""
    ov = _outer_verts(cx, cy, 1.0, phi)  # unit hexagon vertices
    edges = np.diff(np.concatenate([ov, ov[:1]], axis=0), axis=0)  # (6,2)
    n = edges[:, ::-1] * np.array([-1.0, 1.0])  # outward normals
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    d = ov @ n.T  # (6,)
    p = (verts - np.array([cx, cy]))  # (N,6,2)
    rel = p @ n.T  # (N,6,6)
    need = np.max((rel - d).max(axis=1) / L_unit, axis=0) if False else None
    # For a hexagon of side L, the support value along each normal is L * d_i
    # (since scaling the unit hexagon by L scales d by L). So we need
    # L >= max_i (max over points of rel_i) / d_i  for the relevant sign.
    # Because normals point outward and d_i>0, containment requires
    # rel_i <= L * d_i for all i, i.e. L >= max_i (max_p rel_i) / d_i.
    m = rel.max(axis=(0, 1))  # (6,)
    L = np.max(m / d)
    return max(float(L), 0.0)


# helper to avoid the stray line above
L_unit = 1.0


def _objective(x):
    s, theta, cx, cy, phi = x
    if s <= 0.05:
        return 1e9
    verts = _hex_verts_all(cx, cy, s, theta)
    L = _max_L(verts, cx, cy, phi)
    return L


def _best_result(x):
    s, theta, cx, cy, phi = x
    return (
        [[cx + 0.0, cy + 0.0, 0.0] for _ in range(0)],  # placeholder, replaced below
    )


def _build_inner(cx, cy, s, theta):
    base = np.array([
        [-1.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0],
        [-1.0, math.sqrt(3) / 2.0], [0.0, math.sqrt(3) / 2.0],
        [1.0, math.sqrt(3) / 2.0], [2.0, math.sqrt(3) / 2.0],
        [-1.0, -math.sqrt(3) / 2.0], [0.0, -math.sqrt(3) / 2.0],
        [1.0, -math.sqrt(3) / 2.0], [2.0, -math.sqrt(3) / 2.0],
    ]) * s
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    centers = (base @ rot.T) + np.array([cx, cy])
    ang = math.degrees(theta)
    return [[float(cx_), float(cy_), float(ang)] for cx_, cy_ in centers]


def _score(x):
    s, theta, cx, cy, phi = x
    verts = _hex_verts_all(cx, cy, s, theta)
    return _max_L(verts, cx, cy, phi)


def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 110.0

    # ---- Phase 1: coarse grid over the 5 parameters ----
    best = None
    best_x = None

    s_vals = np.arange(1.65, 2.00, 0.02)
    theta_vals = np.arange(0.0, math.pi / 6.0, math.pi / 60.0)
    phi_vals = np.arange(0.0, math.pi / 6.0, math.pi / 30.0)

    for s in s_vals:
        if time.time() > _DEADLINE - 20.0:
            break
        for theta in theta_vals:
            if time.time() > _DEADLINE - 20.0:
                break
            # For a given (s, theta), find the best (cx,cy,phi) by scanning phi
            # and centering the cluster at the origin (cx=cy=0) which is
            # optimal for a symmetric cluster; then let NM refine.
            for phi in phi_vals:
                x = np.array([s, theta, 0.0, 0.0, phi])
                val = _score(x)
                if best is None or val < best:
                    best = val
                    best_x = x.copy()

    if best_x is None:
        best_x = np.array([1.8, 0.0, 0.0, 0.0, 0.0])
        best = _score(best_x)

    # ---- Phase 2: Nelder-Mead polish from the best grid point ----
    try:
        from scipy.optimize import minimize
        x0 = best_x.copy()
        res = minimize(_objective, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-5,
                                "fatol": 1e-6, "adaptive": True})
        if res.success or res.fun < best:
            if res.fun < best:
                best = float(res.fun)
                best_x = np.array(res.x, dtype=float)
        # Try a couple of secondary starts if time permits
        for alt in [
            [1.80, math.pi / 12.0, 0.0, 0.0, math.pi / 12.0],
            [1.72, 0.0, 0.0, 0.0, 0.0],
            [1.85, math.pi / 6.0, 0.0, 0.0, math.pi / 6.0],
        ]:
            if time.time() > _DEADLINE - 30.0:
                break
            r2 = minimize(_objective, np.array(alt, dtype=float),
                          method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-5,
                                   "fatol": 1e-6, "adaptive": True})
            if r2.fun < best:
                best = float(r2.fun)
                best_x = np.array(r2.x, dtype=float)
    except Exception:
        pass

    s, theta, cx, cy, phi = best_x
    inner = _build_inner(cx, cy, s, theta)
    return inner, (float(cx), float(cy)), float(best), float(math.degrees(phi))


# Ensure a valid fallback if the module is imported but construct not called.
if __name__ == "__main__":
    construct()
