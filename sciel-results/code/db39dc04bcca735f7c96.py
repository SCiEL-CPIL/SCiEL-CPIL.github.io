"""Suite task: cover the unit sphere with 50 equal caps (minimize covering radius)."""
import time
import numpy as np
from scipy.spatial import Delaunay

N = 50
_TIME_LIMIT = 150.0
_DEADLINE = time.time() + _TIME_LIMIT - 5.0  # safety margin before kill


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return pts / n


def _covering_radius(pts):
    """Covering radius (radians) = max facet circumradius via Delaunay triangulation."""
    try:
        tri = Delaunay(pts)
    except Exception:
        return np.pi
    # For each triangular facet, compute the circumradius.
    max_r = 0.0
    simplices = tri.simplices
    for idx in simplices:
        a = pts[idx[0]]
        b = pts[idx[1]]
        c = pts[idx[2]]
        # Edge lengths
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)
        # Area via cross product
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        if area < 1e-12:
            continue
        R = (ab * bc * ca) / (4.0 * area)
        if R > max_r:
            max_r = R
    return max_r


def _facet_normal_and_radius(pts, idx):
    a = pts[idx[0]]
    b = pts[idx[1]]
    c = pts[idx[2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-14:
        return None, 0.0
    n = n / nn
    # Make normal point outward (away from origin)
    if np.dot(n, a) < 0:
        n = -n
    ab = np.linalg.norm(a - b)
    bc = np.linalg.norm(b - c)
    ca = np.linalg.norm(c - a)
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    if area < 1e-12:
        return n, 0.0
    R = (ab * bc * ca) / (4.0 * area)
    return n, R


def _worst_facet_refine(pts, steps=4000, base_step=0.02, decay=0.999):
    """Refine by pushing worst-facet vertices outward along the facet normal."""
    pts = pts.copy()
    step = base_step
    best = _covering_radius(pts)
    for _ in range(steps):
        if time.time() > _DEADLINE:
            break
        try:
            tri = Delaunay(pts)
        except Exception:
            break
        simplices = tri.simplices
        # find worst facet
        worst_r = -1.0
        worst_idx = None
        worst_n = None
        for idx in simplices:
            n, R = _facet_normal_and_radius(pts, idx)
            if n is None:
                continue
            if R > worst_r:
                worst_r = R
                worst_idx = idx
                worst_n = n
        if worst_idx is None:
            break
        # push vertices of the worst facet outward along the normal
        for v in worst_idx:
            pts[v] = pts[v] + step * worst_n
        # also gently push the three neighbors? keep simple: just the three vertices
        pts = _normalize(pts)
        r = _covering_radius(pts)
        if r < best - 1e-12:
            best = r
        step *= decay
    return pts, best


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - 5.0

    # Start from a well-spread Fibonacci spiral (not in a hemisphere, no coincidences)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    best = _covering_radius(pts)

    # Worst-facet refinement (shrinks the largest caps)
    pts, best = _worst_facet_refine(pts, steps=3000, base_step=0.015, decay=0.9995)

    # A few local "repulsion + worst-facet" cycles to escape shallow local minima
    for cycle in range(6):
        if time.time() > _DEADLINE:
            break
        # gentle global repulsion to spread points (keeps them off a hemisphere)
        for _ in range(300):
            if time.time() > _DEADLINE:
                break
            force = np.zeros_like(pts)
            for a in range(N):
                for b in range(a + 1, N):
                    d = np.dot(pts[a], pts[b])
                    # repel when too close (d near 1)
                    if d > 0.9:
                        f = (d - 0.9) * 0.05
                        force[a] -= f * pts[b]
                        force[b] -= f * pts[a]
            pts = pts + force
            pts = _normalize(pts)
        # refine again
        pts, r = _worst_facet_refine(pts, steps=1500, base_step=0.01, decay=0.9995)
        if r < best:
            best = r

    pts = _normalize(pts)
    return pts
