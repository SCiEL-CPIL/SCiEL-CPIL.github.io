import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute the 165 triangle index triples
_tri = list(combinations(range(N), 3))
_I = np.array([t[0] for t in _tri])
_J = np.array([t[1] for t in _tri])
_K = np.array([t[2] for t in _tri])

def _min_area(points):
    Pi = points[_I]
    Pj = points[_J]
    Pk = points[_K]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def _in_triangle(points, eps=1e-9):
    x = points[:, 0]
    y = points[:, 1]
    return (y >= -eps) and (y <= np.sqrt(3) * x + eps) and (y <= np.sqrt(3) * (1 - x) + eps)

def _smooth_min_area(points, lam=1.0):
    """Differentiable approximation to the min triangle area (soft-min)."""
    Pi = points[_I]
    Pj = points[_J]
    Pk = points[_K]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    m = areas.min()
    # log-sum-exp soft min: f = -log(sum(exp(-areas/lam)))/lam + m
    shifted = (m - areas) / lam
    logsum = np.log(np.sum(np.exp(shifted)))
    return m - lam * logsum

def _penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    v1 = np.maximum(0.0, -y)
    v2 = np.maximum(0.0, y - np.sqrt(3) * x)
    v3 = np.maximum(0.0, y - np.sqrt(3) * (1 - x))
    return float(np.sum(v1 * v1 + v2 * v2 + v3 * v3))

def _objective(flat, lam=1.0):
    points = flat.reshape(-1, 2)
    area = _smooth_min_area(points, lam)
    pen = _penalty(points)
    return -area + 1000.0 * pen

def construct():
    start = time.time()
    deadline = start + 112.0   # safety margin under 150s

    best_points = None
    best_score = -np.inf

    def consider(points):
        nonlocal best_points, best_score
        if points is None:
            return
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if points.shape != (N, 2):
            return
        if not _in_triangle(points, eps=1e-7):
            return
        s = _min_area(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # ---- Symmetric candidate generator: 2 orbit (3 copies) + 1 center + 6 ring ----
    # Orbit = (cx, cy), (1 - cx, cy), (0.5 + 0.5*(cx - 0.5), H - cy)
    def make_candidate(cx, cy, ring_theta, ring_r):
        cx = float(cx); cy = float(cy)
        ring_theta = np.asarray(ring_theta, dtype=float)
        ring_r = float(ring_r)
        cx2 = 1.0 - cx
        cx3 = 0.5 + 0.5 * (cx - 0.5)
        cy3 = H - cy
        center = np.array([0.5, H / 3.0])
        ring = center + ring_r * np.column_stack([np.cos(ring_theta), np.sin(ring_theta)])
        pts = np.vstack([[cx, cy], [cx2, cy], [cx3, cy3], center, ring])
        return pts

    def valid_ring_theta():
        return np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0

    # ---- Initial diverse candidates ----
    rng = np.random.default_rng(2024)

    # 1. Structured: 3 vertices + center + 7 ring
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, H])
    center = np.array([0.5, H / 3.0])
    ring7 = center + 0.30 * np.column_stack([
        np.cos(np.linspace(0, 2 * np.pi, 7, endpoint=False)),
        np.sin(np.linspace(0, 2 * np.pi, 7, endpoint=False))])
    consider(np.vstack([v1, v2, v3, center, ring7]))

    # 2. 3 vertices + 8 ring (no center)
    ring8 = center + 0.33 * np.column_stack([
        np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)),
        np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))])
    consider(np.vstack([v1, v2, v3, ring8]))

    # 3. Symmetric 2+1+6 candidate
    consider(make_candidate(0.5, 0.30, valid_ring_theta(), 0.22))
    consider(make_candidate(0.35, 0.25, valid_ring_theta(), 0.20))
    consider(make_candidate(0.5, 0.20, valid_ring_theta(), 0.25))

    # 4. Random valid configurations
    for _ in range(6):
        pts = []
        for _ in range(N):
            while True:
                u = rng.random(); v = rng.random()
                if u + v < 1:
                    pts.append([u + 0.5 * v, H * v])
                    break
        consider(np.array(pts))

    if best_points is None:
        # fallback (shouldn't happen)
        best_points = np.vstack([v1, v2, v3, center, ring7])

    # ---- Polish with L-BFGS-B (smooth objective), from multiple starts ----
    starts = [best_points.flatten().copy()]
    # add a couple of symmetric starts as additional basins
    starts.append(make_candidate(0.5, 0.30, valid_ring_theta(), 0.22).flatten())
    starts.append(make_candidate(0.4, 0.25, valid_ring_theta(), 0.20).flatten())
    starts.append(np.vstack([v1, v2, v3, center, ring7]).flatten())

    bounds = [(0.0, 1.0), (0.0, H)] * N

    for s in starts:
        if time.time() > deadline - 8:
            break
        for lam in (1.0, 0.5, 0.25):
            if time.time() > deadline - 8:
                break
            try:
                res = minimize(
                    _objective, s,
                    args=(lam,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
                )
                consider(res.x)
            except Exception:
                pass

    # ---- Local Nelder-Mead on the true (non-smooth) objective for final polish ----
    def _true_obj(flat):
        points = flat.reshape(-1, 2)
        return -_min_area(points) + 1000.0 * _penalty(points)

    def _nm_obj(flat):
        # NM has no bounds; keep points in via a strong penalty
        points = flat.reshape(-1, 2)
        return -_min_area(points) + 5000.0 * _penalty(points)

    if best_points is not None and time.time() < deadline - 6:
        try:
            res = minimize(
                _nm_obj, best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-9},
            )
            consider(res.x)
        except Exception:
            pass

    # ---- Hard projection to guarantee validity ----
    def project_point(x, y):
        if y < 0:
            y = 0.0
        if y > np.sqrt(3) * x:
            dot = x * 0.5 + y * (np.sqrt(3) / 2)
            x = dot * 0.5
            y = dot * (np.sqrt(3) / 2)
        if y > np.sqrt(3) * (1 - x):
            vx = x - 1.0
            vy = y
            dot = vx * 0.5 + vy * (-np.sqrt(3) / 2)
            x = 1.0 + dot * 0.5
            y = 0.0 + dot * (-np.sqrt(3) / 2)
        return [x, y]

    pts = best_points.copy()
    for i in range(N):
        pts[i] = project_point(pts[i, 0], pts[i, 1])

    return pts
