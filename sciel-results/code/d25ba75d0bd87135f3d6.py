"""Sutton-Chen transition-metal cluster global minima - multi-geometry construction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START = time.time()
_HARD_LIMIT = 320.0  # safety margin before the 330 s kill


def _fcc_positions(n):
    """Truncated FCC cluster of n atoms."""
    pts = []
    a = 1.0
    seen = set()
    for i in range(-15, 16):
        for j in range(-15, 16):
            for k in range(-15, 16):
                for (dx, dy, dz) in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    key = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
                    if key in seen:
                        continue
                    seen.add(key)
                    pts.append(p)
    pts = np.array(pts)
    com = pts.mean(axis=0)
    pts = pts - com
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _hcp_positions(n):
    """Truncated HCP cluster of n atoms."""
    a = 1.0
    pts = []
    h = np.sqrt(2.0 / 3.0) * a
    seen = set()
    for layer in range(-8, 9):
        z = layer * h
        for i in range(-8, 9):
            for j in range(-8, 9):
                # hexagonal in-plane lattice
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                p = np.array([x, y, z])
                key = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
                if key in seen:
                    continue
                seen.add(key)
                pts.append(p)
    pts = np.array(pts)
    com = pts.mean(axis=0)
    pts = pts - com
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _spherical_shells(n):
    """Atoms placed on concentric spherical shells (spherical-shell model)."""
    pts = []
    remaining = n
    shell_idx = 0
    while remaining > 0 and shell_idx < 20:
        if shell_idx == 0:
            pts.append(np.zeros(3))
            remaining -= 1
        else:
            # number of points on shell roughly ~ 2*pi*k^2
            count = min(remaining, 2 * 10 * shell_idx * shell_idx)
            radius = 1.0 * np.sqrt(shell_idx)
            # Fibonacci sphere for even distribution
            golden = (1.0 + np.sqrt(5.0)) / 2.0
            for i in range(count):
                z = 1.0 - (2.0 * i + 1.0) / count
                r = np.sqrt(max(0.0, 1.0 - z * z))
                theta = golden * 2.0 * np.pi * i
                p = radius * np.array([r * np.cos(theta), r * np.sin(theta), z])
                pts.append(p)
            remaining -= count
        shell_idx += 1
    pts = np.array(pts)
    com = pts.mean(axis=0)
    pts = pts - com
    return pts[:n]


def _random_compact(n, rng, spacing=0.72):
    """Random compact cluster with minimum separation."""
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    min_sep = 0.55 * spacing
    while len(points) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_sep for q in points):
                points.append(p)
        attempts += 1
    # Fallback: fill any shortfall with jittered points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _build_starts(n, kind, rng):
    """Assemble a list of diverse starting geometries."""
    starts = []
    try:
        starts.append(_fcc_positions(n))
    except Exception:
        pass
    try:
        starts.append(_hcp_positions(n))
    except Exception:
        pass
    try:
        starts.append(_spherical_shells(n))
    except Exception:
        pass
    # several random compact starts
    for _ in range(6):
        try:
            starts.append(_random_compact(n, rng))
        except Exception:
            pass
    # de-duplicate / validate
    valid = []
    for s in starts:
        s = np.asarray(s, dtype=float)
        if s.shape != (n, 3) or not np.all(np.isfinite(s)):
            continue
        # ensure no pair too close
        ok = True
        for i in range(n):
            d = np.linalg.norm(s - s[i], axis=1)
            d[i] = np.inf
            if d.min() < 0.36:
                ok = False
                break
        if ok:
            valid.append(s)
    return valid


def _relax_one(start, potential, kind):
    """Relax a single start; return (energy, positions) or None."""
    try:
        energy, positions = relax(np.asarray(start, dtype=float), potential, kind)
        positions = np.asarray(positions, dtype=float)
        if not np.all(np.isfinite(positions)) or positions.shape != (start.shape):
            return None
        return (float(energy), positions)
    except Exception:
        return None


def construct(instance):
    n = int(instance["n"])
    potential = instance.get("potential", "sutton_chen")
    kind = instance.get("kind")
    rng = np.random.default_rng(12345 + n)

    starts = _build_starts(n, kind, rng)
    if not starts:
        # last-resort fallback
        r = 0.72 * n ** (1.0 / 3.0)
        starts = [rng.uniform(-r, r, (n, 3))]

    def work(s):
        if time.time() - _START > _HARD_LIMIT - 5.0:
            return None
        return _relax_one(s, potential, kind)

    best = None
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(work, starts):
            if res is None:
                continue
            e, pos = res
            if best is None or e < best[0]:
                best = (e, pos)
            if time.time() - _START > _HARD_LIMIT - 10.0:
                break

    if best is None:
        # fallback: single sequential relax
        e, pos = _relax_one(starts[0], potential, kind)
        if e is not None:
            best = (e, pos)
        else:
            return starts[0]

    return best[1]
# EVOLVE-BLOCK-END
