"""Active geometric optimization for covering the sphere with 130 equal caps.

Strategy: start from a Fibonacci spiral, then iteratively repair the largest
empty caps (the hull facets with the biggest spherical circumradius) by moving
points toward the circumcenters, followed by simulated annealing and a local
gradient-free polish.  The covering radius is the score, so we minimize it
directly rather than relying on a generic optimizer.
"""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130
DEADLINE = 120.0          # hard wall-clock limit (s)
SAFETY = 15.0             # return this many seconds before the limit

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return points / norms

def _spherical_circumradius(a, b, c):
    """Circumradius (radians) of the spherical triangle a,b,c (unit vectors).

    The circumcenter is the unit vector equidistant from a, b, c.  It lies in
    the plane spanned by the three points, i.e. in the direction
    (a-b) x (b-c) (or its negative).  We pick the sign facing the centroid.
    """
    v1 = a - b
    v2 = b - c
    x = np.cross(v1, v2)
    nx = np.linalg.norm(x)
    if nx < 1e-12:
        return 0.0
    x = x / nx
    centroid = a + b + c
    if np.dot(x, centroid) < 0:
        x = -x
    cosR = np.clip(np.dot(x, a), -1.0, 1.0)
    return np.arccos(cosR)

def _covering_radius(points):
    """Largest spherical circumradius over the convex-hull facets (radians)."""
    try:
        hull = ConvexHull(points)
        best = 0.0
        for s in hull.simplices:
            r = _spherical_circumradius(points[s[0]], points[s[1]], points[s[2]])
            if r > best:
                best = r
        return best
    except Exception:
        return np.pi  # degenerate -> very bad

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _circumcenter_dir(a, b, c):
    v1 = a - b
    v2 = b - c
    x = np.cross(v1, v2)
    nx = np.linalg.norm(x)
    if nx < 1e-12:
        return None
    x = x / nx
    if np.dot(x, a + b + c) < 0:
        x = -x
    return x

def _greedy_repair(points, deadline):
    """Move the nearest point toward the circumcenter of the largest facet."""
    points = points.copy()
    while time.time() < deadline - SAFETY:
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        best_r = 0.0
        best_s = None
        for s in hull.simplices:
            r = _spherical_circumradius(points[s[0]], points[s[1]], points[s[2]])
            if r > best_r:
                best_r = r
                best_s = s
        if best_s is None:
            break
        a, b, c = points[best_s[0]], points[best_s[1]], points[best_s[2]]
        cc = _circumcenter_dir(a, b, c)
        if cc is None:
            break
        # nearest point to the circumcenter (largest dot)
        dots = points @ cc
        idx = int(np.argmax(dots))
        # move it a fraction of the way toward the circumcenter
        new = points[idx] + 0.5 * (cc - points[idx])
        new = new / np.linalg.norm(new)
        cand = points.copy()
        cand[idx] = new
        cr = _covering_radius(cand)
        if cr < _covering_radius(points):
            points = cand
    return points

def _anneal(points, deadline):
    """Simulated annealing: perturb a random point, accept if it helps (or w/ prob)."""
    rng = np.random.default_rng(12345)
    points = points.copy()
    cur = _covering_radius(points)
    T0 = 0.05
    t_start = time.time()
    step = 0
    while time.time() < deadline - SAFETY:
        idx = rng.integers(N)
        # random small rotation of points[idx]
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        ang = rng.normal(0.0, 0.15)
        # Rodrigues
        k = np.array([0, -axis[2], axis[1], axis[2], 0, -axis[0], -axis[1], axis[0], 0]).reshape(3, 3)
        R = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)
        new = R @ points[idx]
        new = new / np.linalg.norm(new)
        cand = points.copy()
        cand[idx] = new
        new_r = _covering_radius(cand)
        # temperature decays with step
        T = T0 * np.exp(-0.001 * step)
        if new_r < cur or rng.random() < np.exp(-(new_r - cur) / max(T, 1e-9)):
            points = cand
            cur = new_r
        step += 1
    return points

def _polish(points, deadline):
    """Local gradient-free polish on the vertices of the largest facet."""
    points = points.copy()
    rng = np.random.default_rng(999)
    while time.time() < deadline - SAFETY:
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        best_r = 0.0
        best_s = None
        for s in hull.simplices:
            r = _spherical_circumradius(points[s[0]], points[s[1]], points[s[2]])
            if r > best_r:
                best_r = r
                best_s = s
        if best_s is None:
            break
        improved = False
        for vi in best_s:
            base = points[vi]
            for _ in range(6):
                # small random nudge
                nudge = rng.normal(0, 0.03, size=3)
                cand = points.copy()
                cand[vi] = base + nudge
                cand[vi] = cand[vi] / np.linalg.norm(cand[vi])
                cr = _covering_radius(cand)
                if cr < best_r:
                    points = cand
                    best_r = cr
                    improved = True
                    base = cand[vi]
        if not improved:
            break
    return points

def _ensure_valid(points):
    """Remove coincident points and ensure not in a closed hemisphere."""
    points = _normalize(points)
    # separate coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                points[j] = _normalize(points[j] + np.random.default_rng(i * 7 + j).normal(0, 0.05, 3))
    # hemisphere check: if all points lie in some closed hemisphere, nudge
    # (a valid covering of the whole sphere cannot be confined to a hemisphere)
    for _ in range(50):
        # find a direction maximizing the max dot; if max over all points of
        # min dot < 0 then not in hemisphere. Simple test: check if there is a
        # plane through origin with all points on one side.
        ok = False
        for k in range(30):
            d = np.random.default_rng(k).normal(size=3)
            d = d / np.linalg.norm(d)
            if np.all(points @ d >= -1e-12) or np.all(points @ d <= 1e-12):
                ok = True
                break
        if not ok:
            break
        # confined: push one point opposite to its current direction
        i = np.random.default_rng(0).integers(N)
        points[i] = _normalize(-points[i] + np.random.default_rng(i).normal(0, 0.1, 3))
    return points

def construct():
    start = time.time()
    deadline = start + DEADLINE
    points = _fibonacci_sphere(N)
    points = _ensure_valid(points)

    # Phase 1: greedy facet repair
    if time.time() < deadline - SAFETY:
        points = _greedy_repair(points, deadline)

    # Phase 2: simulated annealing
    if time.time() < deadline - SAFETY:
        points = _anneal(points, deadline)

    # Phase 3: local polish
    if time.time() < deadline - SAFETY:
        points = _polish(points, deadline)

    # Final safety: a few more greedy passes if time remains
    if time.time() < deadline - SAFETY:
        points = _greedy_repair(points, deadline)

    points = _ensure_valid(points)
    points = _normalize(points)
    return points
