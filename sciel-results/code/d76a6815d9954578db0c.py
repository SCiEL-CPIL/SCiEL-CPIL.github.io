"""Portfolio bootstrap: structured high-symmetry seeds + local relax + scale/optimize."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 325.0  # wall-clock safety margin below the 330s kill
_GLOBAL_T0 = time.time()
_GLOBAL_DEADLINE = 325.0  # total budget for all construct() calls

def _global_time_left():
    return _GLOBAL_DEADLINE - (time.time() - _GLOBAL_T0)

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_verts():
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n

def _cuboctahedron_verts():
    # 12 vertices: permutations of (±1, ±1, 0)
    v = []
    for x in (-1, 1):
        for y in (-1, 1):
            v.append([x, y, 0.0])
            v.append([x, 0.0, y])
            v.append([0.0, x, y])
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _rhombicubocta_verts():
    # 24 vertices: all permutations of (±1, ±1, ±(1+sqrt2))
    t = 1.0 + 2.0 ** 0.5
    v = []
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                v.append([a, b, c * t])
                v.append([a, b * t, c])
                v.append([a * t, b, c])
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _fcc_lattice(n):
    """Pack n atoms in an FCC-like arrangement, scaled by radius."""
    # Build points on FCC lattice, keep nearest to origin up to n
    R = 2.0
    pts = []
    s = int(R) + 1
    for x in range(-s, s + 1):
        for y in range(-s, s + 1):
            for z in range(-s, s + 1):
                # FCC: points at (0,0,0) type and (0.5,0.5,0) type shifts
                for shift in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]:
                    p = np.array([x + shift[0], y + shift[1], z + shift[2]])
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) == 0:
        return pts
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order]
    if len(pts) > n:
        pts = pts[:n]
    # scale so typical nearest-neighbor distance ~0.72
    if len(pts) >= 2:
        nn = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nn = nn[nn > 1e-6]
        if len(nn):
            scale = 0.72 / np.median(nn)
            pts = pts * scale
    return pts

def _bcc_lattice(n):
    R = 2.0
    pts = []
    s = int(R) + 1
    for x in range(-s, s + 1):
        for y in range(-s, s + 1):
            for z in range(-s, s + 1):
                for shift in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]:
                    p = np.array([x + shift[0], y + shift[1], z + shift[2]])
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) == 0:
        return pts
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order]
    if len(pts) > n:
        pts = pts[:n]
    if len(pts) >= 2:
        nn = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nn = nn[nn > 1e-6]
        if len(nn):
            scale = 0.72 / np.median(nn)
            pts = pts * scale
    return pts

def _random_cluster(size, rng, spacing=0.72, min_dist=0.85):
    radius = spacing * size ** (1.0 / 3.0) * 0.9
    pts = []
    attempts = 0
    while len(pts) < size and attempts < size * size * 40:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
        attempts += 1
    # fallback fill if not enough
    while len(pts) < size:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:size])

def _molten_cluster(size, rng, spacing=0.72, temp=1.5):
    """Higher-temperature random packing for more diverse starts."""
    radius = spacing * size ** (1.0 / 3.0) * 1.2
    pts = []
    attempts = 0
    min_dist = 0.36
    while len(pts) < size and attempts < size * size * 60:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
        attempts += 1
    while len(pts) < size:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:size])

def _deformed_fcc(n, rng, deform=0.3):
    """FCC lattice with random deformation for diversity."""
    base = _fcc_lattice(n)
    if len(base) == 0:
        return base
    base = base.copy()
    # Add random displacement
    base += rng.uniform(-deform, deform, base.shape)
    # Re-center
    base -= base.mean(axis=0)
    # Ensure min distance
    if len(base) >= 2:
        d = np.sqrt(((base[:, None, :] - base[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.36:
            base = base * (0.36 / d.min() + 0.05)
    return base

def _sanitize(positions, min_dist=0.36):
    """Ensure positions are finite and have minimum separation."""
    if not np.all(np.isfinite(positions)):
        return None
    if len(positions) < 2:
        return positions
    d = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    if d.min() < min_dist:
        scale = min_dist / d.min() + 0.05
        positions = positions * scale
    return positions

def _basin_hopping(positions, potential, kind, rng, n_iter=20, sigma=0.15, time_limit=None):
    """Simple basin hopping: perturb, relax, accept if lower energy."""
    best_pos = positions.copy()
    best_e, _ = sutton_chen_energy_gradient(best_pos, kind)
    t_start = time.time()
    for i in range(n_iter):
        if time_limit is not None and (time.time() - t_start) > time_limit:
            break
        # Perturb
        new_pos = best_pos + rng.normal(0, sigma, best_pos.shape)
        # Also try swap perturbation occasionally
        if rng.random() < 0.3 and len(best_pos) >= 2:
            idx = rng.choice(len(best_pos), 2, replace=False)
            new_pos[idx[0]], new_pos[idx[1]] = new_pos[idx[1]].copy(), new_pos[idx[0]].copy()
        new_pos = _sanitize(new_pos)
        if new_pos is None:
            continue
        try:
            e, relaxed = relax(new_pos, potential, kind)
        except Exception:
            continue
        if not np.all(np.isfinite(relaxed)):
            continue
        if e < best_e:
            best_e = float(e)
            best_pos = np.asarray(relaxed, float)
    return best_e, best_pos

def _polish(positions, potential, kind):
    """Final high-precision polish with tight tolerances."""
    try:
        e, pos = relax(positions, potential, kind)
        if np.all(np.isfinite(pos)):
            return float(e), np.asarray(pos, float)
    except Exception:
        pass
    return sutton_chen_energy_gradient(positions, kind)[0], positions

def _shells(n):
    """Yield candidate structured seed arrays for a cluster of n atoms."""
    cands = []
    # 1) pure random pack (few)
    # (handled outside with rng)
    # 2) icosahedral core 12 + shells
    ico = _icosahedron_verts()
    # scale ico to nn ~0.72
    ico = ico * 0.60
    # build by adding shells of icosahedron vertices scaled up
    shells = [ico * (0.60 + 0.55 * k) for k in range(1, 6)]
    # cuboctahedron
    cubo = _cuboctahedron_verts() * 0.60
    # rhombicuboctahedron
    rho = _rhombicubocta_verts() * 0.62
    # FCC & BCC lattice chunks
    cands.append(_fcc_lattice(n))
    cands.append(_bcc_lattice(n))
    # shell combinations: ico core + cubo + rho
    for shell_set in [
        [ico, cubo, rho],
        [ico, rho],
        [ico, cubo],
        [cubo, rho],
        [ico],
        [cubo],
        [rho],
        shells[0], shells[1], shells[2], shells[3],
    ]:
        pts = np.vstack([s for s in shell_set if s is not None and len(s)])
        # dedupe near-identical
        if len(pts) > 1:
            keep = [0]
            for i in range(1, len(pts)):
                if np.min(np.linalg.norm(pts[i] - pts[keep], axis=1)) > 0.3:
                    keep.append(i)
            pts = pts[keep]
        if len(pts) < n:
            continue
        # take first n, but try to center
        pts = pts[:n]
        pts = pts - pts.mean(axis=0)
        cands.append(pts)
    return cands

def construct(instance):
    global _T0
    _T0 = time.time()
    size = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 0.72)
    rng = np.random.default_rng(hash((potential, kind, size)) & 0xFFFFFFFF)

    # Check global time budget
    if _global_time_left() < 5.0:
        # Emergency: return a quick random pack
        return _random_cluster(size, rng, spacing, 0.9)

    # Build candidate seeds
    seeds = []
    # structured
    for c in _shells(size):
        if c is None or len(c) == 0:
            continue
        c = np.asarray(c, float)
        if len(c) != size:
            continue
        c = _sanitize(c)
        if c is not None:
            seeds.append(c)
    # random packs
    for _ in range(4):
        seeds.append(_random_cluster(size, rng, spacing, 0.85))
    # molten packs for diversity
    for _ in range(3):
        seeds.append(_molten_cluster(size, rng, spacing))
    # deformed FCC
    for _ in range(2):
        seeds.append(_deformed_fcc(size, rng, deform=0.25))

    # Phase 1: Relax each seed, keep top candidates
    candidates = []
    for seed in seeds:
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(np.asarray(seed, float), potential, kind)
        except Exception:
            continue
        if not np.all(np.isfinite(positions)):
            continue
        candidates.append((float(energy), np.asarray(positions, float)))
    
    if not candidates:
        # last-ditch random
        for _ in range(10):
            if _time_left() < 6.0:
                break
            seed = _random_cluster(size, rng, spacing, 0.9)
            try:
                energy, positions = relax(seed, potential, kind)
                if np.all(np.isfinite(positions)):
                    candidates.append((float(energy), np.asarray(positions, float)))
            except Exception:
                continue
    
    if not candidates:
        return _random_cluster(size, rng, spacing, 1.0)

    # Sort by energy, keep top 3 for basin hopping
    candidates.sort(key=lambda x: x[0])
    top_seeds = candidates[:3]

    # Phase 2: Basin hopping on top candidates
    bh_time_budget = max(5.0, min(30.0, _time_left() * 0.5))
    best_e, best_pos = top_seeds[0]
    
    for e0, pos0 in top_seeds:
        if _time_left() < 10.0:
            break
        bh_budget = max(2.0, bh_time_budget / len(top_seeds))
        bh_e, bh_pos = _basin_hopping(pos0, potential, kind, rng, n_iter=25, sigma=0.12, time_limit=bh_budget)
        if bh_e < best_e:
            best_e, best_pos = bh_e, bh_pos

    # Phase 3: Try additional diverse starts if time permits
    if _time_left() > 20.0:
        extra_time = _time_left() * 0.4
        for _ in range(5):
            if _time_left() < 10.0:
                break
            seed = _molten_cluster(size, rng, spacing, temp=2.0)
            try:
                e, pos = relax(seed, potential, kind)
                if np.all(np.isfinite(pos)):
                    bh_e, bh_pos = _basin_hopping(np.asarray(pos, float), potential, kind, rng, n_iter=10, sigma=0.1, time_limit=extra_time/5)
                    if bh_e < best_e:
                        best_e, best_pos = bh_e, bh_pos
            except Exception:
                continue

    # Phase 4: Final polish
    if _time_left() > 3.0:
        best_e, best_pos = _polish(best_pos, potential, kind)

    return best_pos
# EVOLVE-BLOCK-END
