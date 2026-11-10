"""Multi-start with parallel relaxation, basin-hopping, and basin merging for Sutton-Chen clusters."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 320.0
_T0 = [time.time()]

def _time_left():
    return _DEADLINE - (time.time() - _T0[0])

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
    R = 3.0
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
    R = 3.0
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
    max_attempts = size * size * 40
    while len(pts) < size and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
        attempts += 1
    while len(pts) < size:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:size])

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

def _relax_one(seed, potential, kind):
    try:
        energy, positions = relax(np.asarray(seed, float), potential, kind)
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            return None
        return (float(energy), positions)
    except Exception:
        return None

def _basin_hopping(pos, potential, kind, rng, n_iter=15, step_scale=0.3):
    """Basin hopping with decreasing step size."""
    best_energy, best_pos = None, None
    try:
        e0, p0 = relax(pos, potential, kind)
        if np.all(np.isfinite(p0)):
            best_energy = float(e0)
            best_pos = np.asarray(p0, float)
    except Exception:
        return pos
    
    size = best_pos.shape[0]
    for it in range(n_iter):
        if _time_left() < 8.0:
            break
        noise = step_scale * (1.0 - 0.4 * it / n_iter)
        perturbed = best_pos + rng.normal(0, noise, size=(size, 3))
        perturbed = _sanitize(perturbed, size)
        if perturbed is None:
            continue
        try:
            energy, positions = relax(perturbed, potential, kind)
            positions = np.asarray(positions, float)
            if np.all(np.isfinite(positions)) and energy < best_energy:
                best_energy = float(energy)
                best_pos = positions
        except Exception:
            continue
    return best_pos

def _merge_basins(pos1, pos2, potential, kind, rng):
    """Try to merge two basins by interpolating and relaxing."""
    size = pos1.shape[0]
    for alpha in [0.3, 0.5, 0.7]:
        if _time_left() < 6.0:
            break
        merged = alpha * pos1 + (1 - alpha) * pos2
        merged = _sanitize(merged, size)
        if merged is None:
            continue
        try:
            energy, positions = relax(merged, potential, kind)
            positions = np.asarray(positions, float)
            if np.all(np.isfinite(positions)):
                return positions
        except Exception:
            continue
    return None

def construct(instance):
    global _T0
    _T0[0] = time.time()
    size = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 0.72)
    rng = np.random.default_rng(hash((potential, kind, size)) & 0xFFFFFFFF)

    # Build candidate seeds
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
    for _ in range(12):
        seeds.append(_random_cluster(size, rng, spacing, 0.85))

    # Parallel relaxation of initial seeds
    best = None
    best_pos = None
    results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_one, seed, potential, kind): seed for seed in seeds}
        for future in as_completed(futures):
            if _time_left() < 20.0:
                break
            result = future.result()
            if result is not None:
                results.append(result)
                energy, positions = result
                if best is None or energy < best[0]:
                    best = (energy, positions)
                    best_pos = positions

    if best is None:
        # Fallback: sequential random relax
        for _ in range(15):
            if _time_left() < 6.0:
                break
            seed = _random_cluster(size, rng, spacing, 0.9)
            try:
                energy, positions = relax(seed, potential, kind)
                if np.all(np.isfinite(positions)):
                    if best is None or energy < best[0]:
                        best = (float(energy), np.asarray(positions, float))
                        best_pos = np.asarray(positions, float)
            except Exception:
                continue

    if best is None:
        pts = _random_cluster(size, rng, spacing, 1.0)
        return np.asarray(pts, float)

    # Sort results by energy
    results.sort(key=lambda x: x[0])
    
    # Basin hopping on the best few structures, in parallel
    top_k = min(3, len(results))
    bh_results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(top_k):
            if _time_left() < 15.0:
                break
            futures.append(executor.submit(_basin_hopping, results[i][1], potential, kind, rng, n_iter=12, step_scale=0.35))
        for future in as_completed(futures):
            if _time_left() < 10.0:
                break
            pos = future.result()
            if pos is not None and np.all(np.isfinite(pos)):
                try:
                    e, p = relax(pos, potential, kind)
                    p = np.asarray(p, float)
                    if np.all(np.isfinite(p)):
                        bh_results.append((float(e), p))
                except Exception:
                    pass
    
    # Update best from basin hopping results
    for energy, positions in bh_results:
        if energy < best[0]:
            best = (energy, positions)
            best_pos = positions

    # Merge basins: try combining top distinct structures
    if len(results) >= 2 and _time_left() > 25.0:
        # Take top 4 distinct structures
        unique_structs = [results[0][1]]
        for i in range(1, min(4, len(results))):
            # Check if structurally distinct (simple RMSD check)
            is_distinct = True
            for us in unique_structs:
                rmsd = np.sqrt(np.mean(np.sum((results[i][1] - us) ** 2, axis=1)))
                if rmsd < 0.1:
                    is_distinct = False
                    break
            if is_distinct:
                unique_structs.append(results[i][1])
        
        # Try merging pairs
        merge_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(len(unique_structs)):
                for j in range(i + 1, len(unique_structs)):
                    if _time_left() < 10.0:
                        break
                    futures.append(executor.submit(_merge_basins, unique_structs[i], unique_structs[j], potential, kind, rng))
            for future in as_completed(futures):
                if _time_left() < 8.0:
                    break
                pos = future.result()
                if pos is not None and np.all(np.isfinite(pos)):
                    try:
                        e, p = relax(pos, potential, kind)
                        p = np.asarray(p, float)
                        if np.all(np.isfinite(p)):
                            merge_results.append((float(e), p))
                    except Exception:
                        pass
        
        for energy, positions in merge_results:
            if energy < best[0]:
                best = (energy, positions)
                best_pos = positions

    # Final perturbation refinement on the best
    best_energy, best_pos = best
    for trial in range(8):
        if _time_left() < 10.0:
            break
        noise_scale = 0.12 * (1.0 - trial / 8.0) + 0.015
        perturbed = best_pos + rng.normal(0, noise_scale, size=(size, 3))
        perturbed = _sanitize(perturbed, size)
        if perturbed is None:
            continue
        try:
            energy, positions = relax(perturbed, potential, kind)
            positions = np.asarray(positions, float)
            if np.all(np.isfinite(positions)) and energy < best_energy:
                best_energy = float(energy)
                best_pos = positions
        except Exception:
            continue

    return best_pos
# EVOLVE-BLOCK-END
