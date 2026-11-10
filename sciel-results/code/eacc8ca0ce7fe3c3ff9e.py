"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_T0 = time.time()
_DEADLINE = 322.0  # safety margin below the 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _sanitize(positions, min_dist=0.36):
    """Ensure all coordinates are finite and no two atoms are closer than min_dist."""
    if not np.all(np.isfinite(positions)):
        mask = ~np.all(np.isfinite(positions), axis=1)
        if np.any(mask):
            positions[mask] = np.random.default_rng(42).uniform(-2, 2, size=(int(np.sum(mask)), 3))
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                if d > 1e-10:
                    direction = (positions[i] - positions[j]) / d
                else:
                    direction = np.array([1.0, 0.0, 0.0])
                positions[i] += direction * (min_dist - d) / 2.0
                positions[j] -= direction * (min_dist - d) / 2.0
    return positions


def _fcc_lattice(n_atoms, spacing=0.72):
    """Generate an FCC-like cluster by cutting a sphere from the FCC lattice."""
    points = []
    a = spacing
    L = int(np.ceil(n_atoms ** (1.0 / 3.0) * 1.5)) + 3
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for fx, fy in [(0, 0), (0, 0.5), (0.5, 0), (0.5, 0.5)]:
                    p = np.array([i + fx, j + fy, k]) * a
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n_atoms]]
    return points


def _bcc_lattice(n_atoms, spacing=0.72):
    """Generate a BCC-like cluster."""
    points = []
    a = spacing
    L = int(np.ceil(n_atoms ** (1.0 / 3.0) * 1.5)) + 3
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                points.append(np.array([i, j, k]) * a)
                points.append(np.array([i + 0.5, j + 0.5, k + 0.5]) * a)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n_atoms]]
    return points


def _random_cluster(n_atoms, spacing=0.72, seed=None):
    rng = np.random.default_rng(seed)
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    min_dist = 0.5 * spacing
    while len(points) < n_atoms and attempts < n_atoms * 50:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n_atoms])


def _generate_seeds(n_atoms, kind, n_seeds=10):
    """Generate diverse initial seeds."""
    spacing = 0.72
    seeds = []
    try:
        seeds.append(_fcc_lattice(n_atoms, spacing))
    except Exception:
        pass
    try:
        seeds.append(_bcc_lattice(n_atoms, spacing))
    except Exception:
        pass
    for s in range(max(0, n_seeds - len(seeds))):
        try:
            seeds.append(_random_cluster(n_atoms, spacing, seed=1000 + s * 7 + n_atoms))
        except Exception:
            pass
    while len(seeds) < 3:
        seeds.append(_random_cluster(n_atoms, spacing, seed=len(seeds) * 13))
    return seeds[:n_seeds]


def _relax_one(positions, potential, kind):
    """Relax a single configuration."""
    try:
        positions = _sanitize(positions)
        energy, pos = relax(positions, potential, kind)
        if np.isfinite(energy):
            pos = _sanitize(pos)
            return (energy, pos)
    except Exception:
        pass
    return (np.inf, None)


def _basin_hopping(positions, potential, kind, n_kicks=5, kick_scale=0.5, rng=None):
    """Basin hopping: perturb and relax."""
    if rng is None:
        rng = np.random.default_rng(hash(tuple(np.round(positions, 6).ravel())) % 100000)
    best_energy, best_pos = _relax_one(positions, potential, kind)
    
    for i in range(n_kicks):
        if _time_left() < 5.0:
            break
        kick = rng.normal(0, kick_scale, size=positions.shape)
        perturbed = positions + kick
        energy, pos = _relax_one(perturbed, potential, kind)
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_pos = pos
            kick_scale *= 0.8
        else:
            kick_scale *= 1.2
            kick_scale = min(kick_scale, 2.0)
    
    return (best_energy, best_pos)


def _constructive_grow(n_atoms, kind, seed_idx=0):
    """
    Coarse-to-fine constructive growth:
    1. Place first atom at origin.
    2. For each subsequent atom, place it at the position that minimizes
       the incremental energy contribution given existing atoms.
    Use a grid search around the existing cluster for candidate positions.
    """
    rng = np.random.default_rng(seed_idx)
    positions = [np.zeros(3)]
    spacing = 0.72
    
    for atom_idx in range(1, n_atoms):
        if _time_left() < 3.0:
            break
        # Determine search radius: grows with cluster size
        radius = spacing * (atom_idx + 1) ** (1.0 / 3.0) * 1.2
        # Grid resolution: coarser for early atoms, finer for later
        n_grid = max(8, min(16, 12 + atom_idx // 5))
        
        best_e = np.inf
        best_pos = None
        
        # Generate candidate positions on a spherical shell + interior
        # Use Fibonacci sphere for even distribution
        n_candidates = 200
        golden = (1 + 5 ** 0.5) / 2
        for m in range(n_candidates):
            z = 1 - 2 * (m + 0.5) / n_candidates
            r_theta = np.sqrt(max(0, 1 - z * z))
            phi = 2 * np.pi * m / golden
            
            # Vary radius
            r_frac = rng.uniform(0.3, 1.0)
            r = radius * r_frac
            
            x = r * r_theta * np.cos(phi)
            y = r * r_theta * np.sin(phi)
            z_coord = r * z
            
            cand = np.array([x, y, z_coord])
            
            # Check minimum distance to existing atoms
            if min(np.linalg.norm(cand - p) for p in positions) < 0.36:
                continue
            
            # Compute incremental energy
            try:
                new_positions = positions + [cand]
                e, _ = sutton_chen_energy_gradient(np.array(new_positions), kind)
                if np.isfinite(e) and e < best_e:
                    best_e = e
                    best_pos = cand
            except Exception:
                continue
        
        if best_pos is None:
            # Fallback: random placement
            for _ in range(20):
                cand = rng.uniform(-radius, radius, 3)
                if min(np.linalg.norm(cand - p) for p in positions) >= 0.36:
                    best_pos = cand
                    break
            if best_pos is None:
                best_pos = rng.uniform(-radius, radius, 3)
        
        positions.append(best_pos)
    
    return np.array(positions[:n_atoms])


def construct(instance):
    """Search for global minimum of Sutton-Chen cluster."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Constructive growth + lattice seeds in parallel
    seeds = _generate_seeds(size, kind, n_seeds=6)
    
    # Add constructive growth seeds
    if _time_left() > 60.0:
        try:
            cg_seed = _constructive_grow(size, kind, seed_idx=42)
            seeds.append(cg_seed)
        except Exception:
            pass
        if _time_left() > 60.0:
            try:
                cg_seed2 = _constructive_grow(size, kind, seed_idx=123)
                seeds.append(cg_seed2)
            except Exception:
                pass
    
    # Phase 1: Parallel relaxation of all seeds
    if _time_left() > 30.0:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_one, s, potential, kind): i for i, s in enumerate(seeds)}
            for future in as_completed(futures):
                if _time_left() < 15.0:
                    break
                try:
                    energy, pos = future.result()
                    if np.isfinite(energy) and energy < best_energy:
                        best_energy = energy
                        best_pos = pos
                except Exception:
                    pass
    
    # Phase 2: Basin hopping from best
    if _time_left() > 20.0 and best_pos is not None:
        rng = np.random.default_rng(777)
        bh_energy, bh_pos = _basin_hopping(best_pos, potential, kind, n_kicks=10, kick_scale=0.4, rng=rng)
        if np.isfinite(bh_energy) and bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Phase 3: Basin hopping from other good seeds
    if _time_left() > 15.0:
        extra_seeds = seeds[1:min(len(seeds), 5)]
        for seed_pos in extra_seeds:
            if _time_left() < 8.0:
                break
            rng = np.random.default_rng(len(extra_seeds) * 13)
            bh_energy, bh_pos = _basin_hopping(seed_pos, potential, kind, n_kicks=5, kick_scale=0.6, rng=rng)
            if np.isfinite(bh_energy) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
    
    # Phase 4: More basin hopping with larger kicks from best
    if _time_left() > 10.0 and best_pos is not None:
        rng = np.random.default_rng(999)
        bh_energy, bh_pos = _basin_hopping(best_pos, potential, kind, n_kicks=5, kick_scale=0.8, rng=rng)
        if np.isfinite(bh_energy) and bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Phase 5: Final polish
    if best_pos is not None and _time_left() > 3.0:
        try:
            best_pos = _sanitize(best_pos)
            energy, pos = relax(best_pos, potential, kind)
            if np.isfinite(energy):
                best_pos = _sanitize(pos)
        except Exception:
            pass
    
    # Fallback
    if best_pos is None:
        best_pos = _random_cluster(size, 0.72, seed=42)
        best_pos = _sanitize(best_pos)
        try:
            _, best_pos = relax(best_pos, potential, kind)
            best_pos = _sanitize(best_pos)
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
