"""Combined: structured seeds + multi-start relax + adaptive basin-hopping from top seeds."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 320.0
_T0 = time.time()

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
    v = []
    for x in (-1, 1):
        for y in (-1, 1):
            v.append([x, y, 0.0])
            v.append([x, 0.0, y])
            v.append([0.0, x, y])
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _rhombicubocta_verts():
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
    R = 2.5
    pts = []
    s = int(R) + 1
    for x in range(-s, s + 1):
        for y in range(-s, s + 1):
            for z in range(-s, s + 1):
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
    if len(pts) >= 2:
        nn = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nn = nn[nn > 1e-6]
        if len(nn):
            scale = 0.72 / np.median(nn)
            pts = pts * scale
    return pts

def _bcc_lattice(n):
    R = 2.5
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
    while len(pts) < size:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:size])

def _molten_cluster(size, rng, spacing=0.72):
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
    base += rng.uniform(-deform, deform, base.shape)
    base -= base.mean(axis=0)
    if len(base) >= 2:
        d = np.sqrt(((base[:, None, :] - base[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.36:
            base = base * (0.36 / d.min() + 0.05)
    return base

def _shells(n):
    cands = []
    ico = _icosahedron_verts() * 0.60
    cubo = _cuboctahedron_verts() * 0.60
    rho = _rhombicubocta_verts() * 0.62
    shells = [ico * (0.60 + 0.55 * k) for k in range(1, 6)]
    cands.append(_fcc_lattice(n))
    cands.append(_bcc_lattice(n))
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
        if len(pts) > 1:
            keep = [0]
            for i in range(1, len(pts)):
                if np.min(np.linalg.norm(pts[i] - pts[keep], axis=1)) > 0.3:
                    keep.append(i)
            pts = pts[keep]
        if len(pts) < n:
            continue
        pts = pts[:n]
        pts = pts - pts.mean(axis=0)
        cands.append(pts)
    return cands

def _sanitize(pos, size):
    if not np.all(np.isfinite(pos)):
        return None
    if size < 2:
        return pos
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    if d.min() < 0.35:
        scale = 0.35 / d.min() + 0.05
        pos = pos * scale
    return pos

def _adaptive_basin_hopping(pos, potential, kind, rng, n_iter=25, max_step=0.45, min_step=0.05):
    """Adaptive basin-hopping: large steps early, small steps late, with occasional restarts."""
    best_energy, best_pos = None, None
    try:
        e0, p0 = relax(pos, potential, kind)
        if np.all(np.isfinite(p0)):
            best_energy = float(e0)
            best_pos = np.asarray(p0, float)
    except Exception:
        return pos

    for it in range(n_iter):
        if _time_left() < 10.0:
            break
        frac = it / max(n_iter - 1, 1)
        step = max_step * (1.0 - frac) + min_step * frac
        if it > 0 and it % 7 == 0:
            step *= 1.8
        
        perturbed = best_pos + rng.normal(0, step, size=(best_pos.shape[0], 3))
        perturbed = _sanitize(perturbed, best_pos.shape[0])
        if perturbed is None:
            continue
        try:
            energy, positions = relax(perturbed, potential, kind)
        except Exception:
            continue
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            continue
        if best_energy is None or energy < best_energy:
            best_energy = float(energy)
            best_pos = positions
    return best_pos

def _swap_basin_hopping(pos, potential, kind, rng, n_iter=15):
    """Basin hopping with swap moves for better exploration."""
    best_energy, best_pos = None, None
    try:
        e0, p0 = relax(pos, potential, kind)
        if np.all(np.isfinite(p0)):
            best_energy = float(e0)
            best_pos = np.asarray(p0, float)
    except Exception:
        return pos

    n_atoms = best_pos.shape[0]
    for it in range(n_iter):
        if _time_left() < 10.0:
            break
        perturbed = best_pos.copy()
        if rng.random() < 0.5 and n_atoms >= 2:
            idx = rng.choice(n_atoms, 2, replace=False)
            perturbed[idx[0]], perturbed[idx[1]] = perturbed[idx[1]].copy(), perturbed[idx[0]].copy()
        else:
            step = 0.3 * (1.0 - it / max(n_iter - 1, 1)) + 0.05
            perturbed = best_pos + rng.normal(0, step, size=(n_atoms, 3))
        
        perturbed = _sanitize(perturbed, n_atoms)
        if perturbed is None:
            continue
        try:
            energy, positions = relax(perturbed, potential, kind)
        except Exception:
            continue
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            continue
        if best_energy is None or energy < best_energy:
            best_energy = float(energy)
            best_pos = positions
    return best_pos

def _temperature_basin_hopping(pos, potential, kind, rng, n_iter=20, t_init=0.15, t_min=0.02):
    """Metropolis-style basin hopping with temperature annealing."""
    best_energy, best_pos = None, None
    try:
        e0, p0 = relax(pos, potential, kind)
        if np.all(np.isfinite(p0)):
            best_energy = float(e0)
            best_pos = np.asarray(p0, float)
    except Exception:
        return pos

    current_energy = best_energy
    current_pos = best_pos
    n_atoms = best_pos.shape[0]
    
    for it in range(n_iter):
        if _time_left() < 10.0:
            break
        T = t_init * (t_min / t_init) ** (it / max(n_iter - 1, 1))
        step = 0.15 + 0.1 * (1.0 - it / max(n_iter - 1, 1))
        
        perturbed = current_pos + rng.normal(0, step, size=(n_atoms, 3))
        if rng.random() < 0.3 and n_atoms >= 2:
            idx = rng.choice(n_atoms, 2, replace=False)
            perturbed[idx[0]], perturbed[idx[1]] = perturbed[idx[1]].copy(), perturbed[idx[0]].copy()
        
        perturbed = _sanitize(perturbed, n_atoms)
        if perturbed is None:
            continue
        try:
            energy, positions = relax(perturbed, potential, kind)
        except Exception:
            continue
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            continue
        
        # Metropolis acceptance
        delta_e = float(energy) - current_energy
        if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
            current_energy = float(energy)
            current_pos = positions
            if best_energy is None or current_energy < best_energy:
                best_energy = current_energy
                best_pos = positions
    return best_pos

def construct(instance):
    global _T0
    _T0 = time.time()
    size = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 0.72)
    rng = np.random.default_rng(hash((potential, kind, size)) & 0xFFFFFFFF)

    # Build diverse candidate seeds
    seeds = []
    for c in _shells(size):
        if c is None or len(c) == 0:
            continue
        c = np.asarray(c, float)
        if len(c) != size:
            continue
        c = _sanitize(c, size)
        if c is not None:
            seeds.append(c)
    
    # More random seeds with varied spacing and min_dist
    for sp in [0.65, 0.72, 0.80, 0.90]:
        for md in [0.80, 0.90]:
            seeds.append(_random_cluster(size, rng, sp, md))
    
    # Add molten and deformed FCC seeds
    seeds.append(_molten_cluster(size, rng, spacing))
    seeds.append(_deformed_fcc(size, rng, 0.3))
    seeds.append(_deformed_fcc(size, rng, 0.5))

    # Relax all seeds, keep top 6
    results = []
    for seed in seeds:
        if _time_left() < 20.0:
            break
        try:
            energy, positions = relax(np.asarray(seed, float), potential, kind)
        except Exception:
            continue
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            continue
        results.append((float(energy), positions))

    if not results:
        for _ in range(10):
            if _time_left() < 6.0:
                break
            seed = _random_cluster(size, rng, spacing, 0.9)
            try:
                energy, positions = relax(seed, potential, kind)
                if np.all(np.isfinite(positions)):
                    results.append((float(energy), np.asarray(positions, float)))
            except Exception:
                continue

    if not results:
        pts = _random_cluster(size, rng, spacing, 1.0)
        return np.asarray(pts, float)

    # Sort by energy, take top 6 distinct structures for basin hopping
    results.sort(key=lambda x: x[0])
    top_seeds = []
    for e, p in results:
        if len(top_seeds) >= 6:
            break
        is_distinct = True
        for _, existing in top_seeds:
            d = np.linalg.norm(p - existing, axis=1)
            if np.median(d) < 0.3:
                is_distinct = False
                break
        if is_distinct:
            top_seeds.append((e, p))

    # Run basin-hopping from each top seed with different strategies
    best_energy, best_pos = top_seeds[0]
    
    for i, (seed_e, seed_p) in enumerate(top_seeds):
        if _time_left() < 15.0:
            break
        # Alternate between adaptive BH, swap BH, and temperature BH
        if i % 3 == 0:
            n_iter = 25 if i == 0 else 15
            bh_result = _adaptive_basin_hopping(seed_p, potential, kind, rng, n_iter=n_iter, max_step=0.45, min_step=0.05)
        elif i % 3 == 1:
            bh_result = _swap_basin_hopping(seed_p, potential, kind, rng, n_iter=15)
        else:
            bh_result = _temperature_basin_hopping(seed_p, potential, kind, rng, n_iter=20, t_init=0.15, t_min=0.02)
        
        if bh_result is not None and np.all(np.isfinite(bh_result)):
            try:
                e_bh, _ = sutton_chen_energy_gradient(bh_result, kind)
                if np.isfinite(e_bh) and e_bh < best_energy:
                    best_energy = float(e_bh)
                    best_pos = bh_result
            except Exception:
                pass

    # Final refinement: temperature BH from the current best
    if _time_left() > 20.0:
        bh_final = _temperature_basin_hopping(best_pos, potential, kind, rng, n_iter=15, t_init=0.10, t_min=0.01)
        if bh_final is not None and np.all(np.isfinite(bh_final)):
            try:
                e_final, _ = sutton_chen_energy_gradient(bh_final, kind)
                if np.isfinite(e_final) and e_final < best_energy:
                    best_energy = float(e_final)
                    best_pos = bh_final
            except Exception:
                pass

    # Final tight relaxation
    if _time_left() > 5.0:
        try:
            e_polish, p_polish = relax(best_pos, potential, kind)
            p_polish = np.asarray(p_polish, float)
            if np.all(np.isfinite(p_polish)) and e_polish < best_energy:
                best_energy = float(e_polish)
                best_pos = p_polish
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
