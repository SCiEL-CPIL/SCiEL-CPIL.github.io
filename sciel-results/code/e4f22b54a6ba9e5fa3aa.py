"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 300.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    pts.append(np.array([i+dx, j+dy, k+dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_bcc(n):
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append(np.array([i, j, k]))
                pts.append(np.array([i+0.5, j+0.5, k+0.5]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _make_icosahedral(n):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1,0,phi],[-1,0,phi],[1,0,-phi],[-1,0,-phi],
        [0,phi,1],[0,-phi,1],[0,phi,-1],[0,-phi,-1],
        [phi,1,0],[-phi,1,0],[phi,-1,0],[-phi,-1,0]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    radii = [1.0, 1.618, 2.236, 2.828, 3.317]
    for r in radii:
        for v in verts:
            pts.append(r * v)
        for i in range(12):
            for j in range(i+1, 12):
                mid = (verts[i] + verts[j]) / 2
                mid /= np.linalg.norm(mid)
                pts.append(r * mid)
    pts = np.array(pts)
    unique_pts = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > 0.01 for q in unique_pts):
            unique_pts.append(p)
    pts = np.array(unique_pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts *= (0.72 * n**(1.0/3.0) * 0.6) / rmax
    return pts


def _perturb(positions, rng, scale=0.1):
    return positions + rng.normal(0, scale, positions.shape)


def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345)
    best_energy = np.inf
    best_positions = None
    starts = []

    try:
        fcc = _make_fcc(n)
        if _is_valid(fcc):
            starts.append(fcc)
        for s in [0.05, 0.1, 0.2]:
            pert = _perturb(fcc, rng, s)
            if _is_valid(pert):
                starts.append(pert)
    except Exception:
        pass

    try:
        bcc = _make_bcc(n)
        if _is_valid(bcc):
            starts.append(bcc)
        for s in [0.05, 0.1]:
            pert = _perturb(bcc, rng, s)
            if _is_valid(pert):
                starts.append(pert)
    except Exception:
        pass

    try:
        ico = _make_icosahedral(n)
        if _is_valid(ico):
            starts.append(ico)
        for s in [0.05, 0.1]:
            pert = _perturb(ico, rng, s)
            if _is_valid(pert):
                starts.append(pert)
    except Exception:
        pass

    spacing = 0.72
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    for _ in range(3):
        points = []
        attempts = 0
        while len(points) < n and attempts < 5000:
            attempts += 1
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                    points.append(p)
        if len(points) == n:
            arr = np.array(points)
            if _is_valid(arr):
                starts.append(arr)

    for i, start in enumerate(starts):
        if _time_left() < 10.0:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    if best_positions is None:
        try:
            best_positions = _make_fcc(n)
        except Exception:
            best_positions = rng.uniform(-1, 1, (n, 3))

    return best_positions
# EVOLVE-BLOCK-END
