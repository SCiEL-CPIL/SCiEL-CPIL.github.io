"""
Sutton-Chen transition-metal cluster global minima: multi-start from diverse
structured + jittered compact seeds, each relaxed with L-BFGS-B, under a strict
wall-clock deadline. Returns the (n,3) coordinates of the best relaxed cluster.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax


# ---- hard wall-clock deadline (evaluator kills at 330 s; keep a safety margin) ----
_DEADLINE_SECONDS = 315.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


# ---------------------------------------------------------------------------
# Seed generators: produce compact, valid (min-pair-distance) (n,3) arrays.
# All are centered near the origin and scaled so typical nearest-neighbor
# distances are ~ the equilibrium spacing of the potential.
# ---------------------------------------------------------------------------

def _fcc_seed(n, rng, scale=1.0):
    """Fill the fcc lattice with the n sites closest to the origin."""
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    cand = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cell = np.array([i, j, k])
                for b in basis:
                    p = (cell + b)
                    cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    # center
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _hcp_seed(n, rng, scale=1.0):
    """Fill an hcp-like lattice (two interpenetrating triangular layers)."""
    a = 1.0
    h = np.sqrt(2.0 / 3.0) * a  # layer spacing for hcp
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    cand = []
    for layer in range(-max_shell, max_shell + 1):
        z = layer * h
        is_b = (layer % 2 != 0)
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * j)
                y = a * (np.sqrt(3.0) / 2.0) * j
                if is_b:
                    x += a * 0.5
                    y += a * np.sqrt(3.0) / 6.0
                p = np.array([x, y, z])
                cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _icosahedral_seed(n, rng, scale=1.0):
    """Start from an icosahedron and add shells / fill interior by nearest
    available point on a fine icosahedral-fcc hybrid. Fallback: fcc."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    pts_list = [p * 1.0 for p in ico]
    # add fcc points to fill up to n
    fcc = _fcc_seed(n, rng, scale=1.0)
    all_pts = list(pts_list)
    for p in fcc:
        if len(all_pts) >= n:
            break
        ok = True
        for q in all_pts:
            if np.linalg.norm(p - q) < 0.45:
                ok = False
                break
        if ok:
            all_pts.append(p)
    if len(all_pts) < n:
        for p in fcc:
            if len(all_pts) >= n:
                break
            all_pts.append(p)
    pts = np.array(all_pts[:n], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _random_compact_seed(n, rng, scale=1.0, min_dist=0.55):
    """Random points in a ball with a minimum-pair-distance constraint."""
    radius = scale * n ** (1.0 / 3.0) * 0.85
    pts = []
    tries = 0
    max_tries = 20000
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    if len(pts) < n:
        md = min_dist
        while len(pts) < n:
            md *= 0.95
            if md < 0.36:
                while len(pts) < n:
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius:
                        pts.append(p)
                break
            tries = 0
            while len(pts) < n and tries < max_tries:
                tries += 1
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if all(np.linalg.norm(p - q) >= md for q in pts):
                    pts.append(p)
    pts = np.array(pts, dtype=float)
    pts = pts - pts.mean(axis=0)
    return pts


def _jitter(seed, rng, amount):
    return seed + rng.normal(0.0, amount, seed.shape)


def _is_valid(pts, min_dist=0.35):
    if not np.all(np.isfinite(pts)):
        return False
    n = len(pts)
    if n < 2:
        return True
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    return dist.min() >= min_dist


def _try_relax(seed, potential, kind):
    """Relax a seed; return (energy, positions) or (inf, None) on failure."""
    try:
        energy, positions = relax(seed, potential, kind)
    except Exception:
        return np.inf, None
    if not np.all(np.isfinite(positions)):
        return np.inf, None
    if not _is_valid(positions):
        return np.inf, None
    return float(energy), positions


def construct(instance):
    global _START_TIME
    _START_TIME = time.time()

    n = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")

    base_scale = 1.0

    rng = np.random.default_rng(12345 + n)

    seeds = []

    # 1. Pure structured seeds
    for gen in (_fcc_seed, _hcp_seed, _icosahedral_seed):
        s = gen(n, rng, scale=base_scale)
        if _is_valid(s, 0.4):
            seeds.append(s)
        for amt in (0.05, 0.12, 0.25):
            sj = _jitter(s, rng, amt)
            if _is_valid(sj, 0.36):
                seeds.append(sj)

    # 2. Random compact seeds
    for _ in range(6):
        s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.6)
        if _is_valid(s, 0.36):
            seeds.append(s)

    best_energy = np.inf
    best_positions = None

    for s in seeds:
        if _remaining() < 5.0:
            break
        e, pos = _try_relax(s, potential, kind)
        if e < best_energy:
            best_energy = e
            best_positions = pos

    if best_positions is None:
        # Fallback
        pos = []
        rng2 = np.random.default_rng(0)
        while len(pos) < n:
            p = rng2.uniform(-5, 5, 3)
            if all(np.linalg.norm(p - q) >= 0.8 for q in pos):
                pos.append(p)
        best_positions = np.array(pos[:n])

    return best_positions
# EVOLVE-BLOCK-END
