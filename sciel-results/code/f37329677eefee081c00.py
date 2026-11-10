"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_T0 = time.time()
_DEADLINE = 318.0  # safety margin below the 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _sanitize(positions, min_dist=0.36):
    """Ensure all coordinates are finite and no two atoms are closer than min_dist."""
    positions = np.array(positions, dtype=float)
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


def _hcp_lattice(n_atoms, spacing=0.72):
    """Generate an HCP-like cluster."""
    points = []
    a = spacing
    c_over_a = 1.633
    L = int(np.ceil(n_atoms ** (1.0 / 3.0) * 1.5)) + 3
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3) / 2.0
                z = k * c_over_a
                points.append(np.array([x, y, z]) * a)
                if k % 2 == 1:
                    points.append(np.array([x + 1.0 / 3.0, y + np.sqrt(3) / 6.0, z]) * a)
                    points.append(np.array([x + 2.0 / 3.0, y - np.sqrt(3) / 6.0, z]) * a)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n_atoms]]
    return points


def _icosahedron_shell(n_atoms, spacing=0.72):
    """Generate icosahedral shell clusters."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    # Inner icosahedron
    for v in verts:
        points.append(v * spacing * 0.9)
    # Outer icosahedron
    for v in verts:
        points.append(v * spacing * 1.8)
    # More points to fill
    rng = np.random.default_rng(123)
    while len(points) < n_atoms:
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        r = spacing * (0.9 + rng.uniform(0, 1.0))
        points.append(v * r)
    return np.array(points[:n_atoms])


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


def _shells(n_atoms, spacing=0.72, seed=None):
    """Generate a cluster with concentric shells."""
    rng = np.random.default_rng(seed)
    points = [np.zeros(3)]
    shell_radii = [spacing * 0.9, spacing * 1.6, spacing * 2.3, spacing * 3.0]
    for r in shell_radii:
        if len(points) >= n_atoms:
            break
        # Place points on sphere of radius r
        max_on_shell = int(4 * np.pi * r * r / (spacing * spacing * 0.5))
        for _ in range(max_on_shell):
            if len(points) >= n_atoms:
                break
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v)
            p = v * r
            if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in points):
                points.append(p)
    while len(points) < n_atoms:
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        r = spacing * (0.9 + rng.uniform(0, 2.5))
        points.append(v * r)
    return np.array(points[:n_atoms])


def _molten_cluster(n_atoms, spacing=0.72, seed=None):
    """Generate a molten-like cluster with high-temperature-like disorder."""
    rng = np.random.default_rng(seed)
    radius = spacing * n_atoms ** (1.0 / 3.0) * 1.0
    points = []
    for _ in range(n_atoms):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        r = rng.uniform(0, radius)
        points.append(v * r)
    return np.array(points[:n_atoms])


def _deformed_fcc(n_atoms, spacing=0.72, seed=None):
    """Generate a deformed FCC cluster."""
    rng = np.random.default_rng(seed)
    base = _fcc_lattice(n_atoms, spacing)
    noise = rng.normal(0, 0.15, size=base.shape)
    return base + noise


def _generate_seeds(n_atoms, kind, n_seeds=12):
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
    try:
        seeds.append(_hcp_lattice(n_atoms, spacing))
    except Exception:
        pass
    try:
        seeds.append(_icosahedron_shell(n_atoms, spacing))
    except Exception:
        pass
    try:
        seeds.append(_shells(n_atoms, spacing, seed=42))
    except Exception:
        pass
    try:
        seeds.append(_molten_cluster(n_atoms, spacing, seed=99))
    except Exception:
        pass
    try:
        seeds.append(_deformed_fcc(n_atoms, spacing, seed=7))
    except Exception:
        pass
    for s in range(max(0, n_seeds - len(seeds))):
        try:
            seeds.append(_random_cluster(n_atoms, spacing, seed=1000 + s * 7 + n_atoms))
        except Exception:
            pass
    while len(seeds) < 4:
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


def _basin_hopping(positions, potential, kind, n_kicks=15, kick_scale=0.4, rng=None):
    """Basin hopping with adaptive kick scale."""
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
            kick_scale *= 0.85
        else:
            kick_scale *= 1.15
            kick_scale = min(kick_scale, 2.5)
    
    return (best_energy, best_pos)


def _adaptive_basin_hopping(positions, potential, kind, n_kicks=20, rng=None):
    """Adaptive basin hopping with varying kick directions and scales."""
    if rng is None:
        rng = np.random.default_rng(hash(tuple(np.round(positions, 6).ravel())) % 100000)
    best_energy, best_pos = _relax_one(positions, potential, kind)
    kick_scale = 0.3
    
    for i in range(n_kicks):
        if _time_left() < 4.0:
            break
        # Vary kick scale periodically
        current_scale = kick_scale * (1.0 + 0.3 * np.sin(i * 0.5))
        kick = rng.normal(0, current_scale, size=positions.shape)
        perturbed = positions + kick
        energy, pos = _relax_one(perturbed, potential, kind)
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_pos = pos
            kick_scale *= 0.9
        else:
            kick_scale *= 1.1
            kick_scale = min(kick_scale, 3.0)
    
    return (best_energy, best_pos)


def _polish(positions, potential, kind):
    """Final high-precision polish with multiple relaxation passes."""
    try:
        positions = _sanitize(positions)
        for _ in range(3):
            if _time_left() < 3.0:
                break
            energy, pos = relax(positions, potential, kind)
            if np.isfinite(energy):
                positions = _sanitize(pos)
        return positions
    except Exception:
        return positions


def construct(instance):
    """Search for global minimum of Sutton-Chen cluster."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse seeds
    seeds = _generate_seeds(size, kind, n_seeds=12)
    
    # Phase 1: Parallel relaxation of all seeds
    if _time_left() > 30.0:
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_one, s, potential, kind): i for i, s in enumerate(seeds)}
            for future in as_completed(futures):
                if _time_left() < 15.0:
                    break
                try:
                    energy, pos = future.result()
                    results.append((energy, pos))
                    if np.isfinite(energy) and energy < best_energy:
                        best_energy = energy
                        best_pos = pos
                except Exception:
                    pass
    
    # Sort results by energy
    valid_results = [(e, p) for e, p in results if np.isfinite(e) and p is not None]
    valid_results.sort(key=lambda x: x[0])
    
    # Phase 2: Basin hopping from top seeds (parallel where possible)
    if _time_left() > 25.0 and best_pos is not None:
        # Basin hop from the best seed
        rng = np.random.default_rng(777)
        bh_energy, bh_pos = _adaptive_basin_hopping(best_pos, potential, kind, n_kicks=20, rng=rng)
        if np.isfinite(bh_energy) and bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Basin hop from other good seeds
    if _time_left() > 20.0:
        extra_seeds = valid_results[1:min(len(valid_results), 5)]
        for seed_pos in extra_seeds:
            if _time_left() < 8.0:
                break
            rng = np.random.default_rng(len(extra_seeds) * 13 + 7)
            bh_energy, bh_pos = _adaptive_basin_hopping(seed_pos, potential, kind, n_kicks=10, rng=rng)
            if np.isfinite(bh_energy) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
    
    # Phase 3: More aggressive basin hopping from best with larger kicks
    if _time_left() > 15.0 and best_pos is not None:
        rng = np.random.default_rng(999)
        bh_energy, bh_pos = _adaptive_basin_hopping(best_pos, potential, kind, n_kicks=15, rng=rng)
        if np.isfinite(bh_energy) and bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Phase 4: Second round of basin hopping from multiple seeds in parallel
    if _time_left() > 20.0 and best_pos is not None:
        n_extra = min(3, max(1, int(_time_left() // 10)))
        extra_bh_seeds = []
        rng_master = np.random.default_rng(5555)
        for i in range(n_extra):
            kick = rng_master.normal(0, 0.8, size=best_pos.shape)
            extra_bh_seeds.append(best_pos + kick)
        
        if _time_left() > 10.0:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_adaptive_basin_hopping, s, potential, kind, 8, 
                           np.random.default_rng(i * 100 + 3)): i for i, s in enumerate(extra_bh_seeds)}
                for future in as_completed(futures):
                    if _time_left() < 5.0:
                        break
                    try:
                        bh_energy, bh_pos = future.result()
                        if np.isfinite(bh_energy) and bh_energy < best_energy:
                            best_energy = bh_energy
                            best_pos = bh_pos
                    except Exception:
                        pass
    
    # Phase 5: Final polish
    if best_pos is not None and _time_left() > 3.0:
        best_pos = _polish(best_pos, potential, kind)
        try:
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
