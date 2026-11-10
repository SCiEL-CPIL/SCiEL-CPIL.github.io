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
        # Replace non-finite with random values
        mask = ~np.all(np.isfinite(positions), axis=1)
        if np.any(mask):
            positions[mask] = np.random.default_rng(42).uniform(-2, 2, size=(np.sum(mask), 3))
    # Check pairwise distances
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                # Separate them
                if d > 1e-10:
                    direction = (positions[i] - positions[j]) / d
                else:
                    direction = np.array([1.0, 0.0, 0.0])
                positions[i] += direction * (min_dist - d) / 2.0
                positions[j] -= direction * (min_dist - d) / 2.0
    return positions


def _icosahedron_verts():
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n


def _cuboctahedron_verts():
    v = np.array([
        [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
        [0.0, 1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, -1.0, -1.0],
        [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0],
    ], dtype=float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n


def _fcc_lattice(n_atoms, spacing=0.72):
    """Generate an FCC-like cluster by cutting a sphere from the FCC lattice."""
    points = []
    # FCC lattice vectors
    a = spacing
    # Generate points in a bounding box
    L = int(np.ceil(n_atoms ** (1.0 / 3.0) * 1.5)) + 3
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
                for fx, fy in [(0, 0), (0, 0.5), (0.5, 0), (0.5, 0.5)]:
                    p = np.array([i + fx, j + fy, k]) * a
                    points.append(p)
    points = np.array(points)
    # Sort by distance from origin and take first n_atoms
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
                # BCC: (0,0,0) and (0.5,0.5,0.5) * a
                points.append(np.array([i, j, k]) * a)
                points.append(np.array([i + 0.5, j + 0.5, k + 0.5]) * a)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n_atoms]]
    return points


def _shells(n_atoms, spacing=0.72):
    """Build a cluster from concentric icosahedral shells."""
    ico = _icosahedron_verts()
    points = [np.zeros(3)]  # center
    shell_radii = [1.0, 1.6, 2.1, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]
    shell_counts = [12, 42, 92, 162, 252, 362, 492, 642, 812, 1002]
    
    for idx, (r, max_count) in enumerate(zip(shell_radii, shell_counts)):
        if len(points) >= n_atoms:
            break
        # Use icosahedron vertices scaled
        shell_points = ico * r * spacing
        # Add more points by rotating
        while len(points) < n_atoms and len(shell_points) < max_count:
            # Add rotated versions
            angle = 0.3 * (len(points) + idx)
            rot = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            new_pts = (shell_points @ rot.T)
            shell_points = np.vstack([shell_points, new_pts])
        
        remaining = n_atoms - len(points)
        take = min(remaining, len(shell_points))
        if take > 0:
            points.extend(shell_points[:take].tolist())
    
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
    # If we couldn't place all, just add random points
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n_atoms])


def _generate_seeds(n_atoms, kind, n_seeds=10):
    """Generate diverse initial seeds."""
    spacing = 0.72
    seeds = []
    
    # Structured seeds
    try:
        seeds.append(_fcc_lattice(n_atoms, spacing))
    except Exception:
        pass
    try:
        seeds.append(_bcc_lattice(n_atoms, spacing))
    except Exception:
        pass
    try:
        seeds.append(_shells(n_atoms, spacing))
    except Exception:
        pass
    
    # Icosahedron-based
    try:
        ico = _icosahedron_verts()
        if n_atoms <= 13:
            seeds.append(np.vstack([np.zeros(3), ico * spacing]))
        else:
            # Two shells
            inner = np.vstack([np.zeros(3), ico * spacing])
            outer = ico * spacing * 1.6
            # Add rotated outer
            angle = 0.5
            rot = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            outer2 = (outer @ rot.T)
            all_pts = np.vstack([inner, outer, outer2])
            # Take closest to origin
            dists = np.linalg.norm(all_pts, axis=1)
            order = np.argsort(dists)
            seeds.append(all_pts[order[:n_atoms]])
    except Exception:
        pass
    
    # Random seeds with different seeds
    for s in range(max(0, n_seeds - len(seeds))):
        try:
            seeds.append(_random_cluster(n_atoms, spacing, seed=1000 + s * 7 + n_atoms))
        except Exception:
            pass
    
    # Ensure we have at least a few seeds
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


def _basin_hopping(positions, potential, kind, n_kicks=5, kick_scale=0.5):
    """Simple basin hopping: perturb and relax."""
    best_energy, best_pos = _relax_one(positions, potential, kind)
    rng = np.random.default_rng(hash(tuple(positions.ravel())) % 100000)
    
    for i in range(n_kicks):
        if _time_left() < 5.0:
            break
        # Perturb
        kick = rng.normal(0, kick_scale, size=positions.shape)
        perturbed = positions + kick
        energy, pos = _relax_one(perturbed, potential, kind)
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_pos = pos
            # Reduce kick scale on success
            kick_scale *= 0.8
        else:
            # Increase kick scale slightly on failure
            kick_scale *= 1.2
            kick_scale = min(kick_scale, 2.0)
    
    return (best_energy, best_pos)


def _polish(positions, potential, kind):
    """Final polish with tighter tolerances - just do another relax."""
    try:
        positions = _sanitize(positions)
        energy, pos = relax(positions, potential, kind)
        if np.isfinite(energy):
            return _sanitize(pos)
    except Exception:
        pass
    return positions


def construct(instance):
    """Search for global minimum of Sutton-Chen cluster."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Generate diverse seeds
    seeds = _generate_seeds(size, kind, n_seeds=8)
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Parallel relaxation of initial seeds
    if _time_left() > 30.0:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_one, s, potential, kind): i for i, s in enumerate(seeds)}
            for future in as_completed(futures):
                if _time_left() < 10.0:
                    break
                try:
                    energy, pos = future.result()
                    if np.isfinite(energy) and energy < best_energy:
                        best_energy = energy
                        best_pos = pos
                except Exception:
                    pass
    
    # Phase 2: Basin hopping from best configurations
    if _time_left() > 20.0 and best_pos is not None:
        # Do basin hopping from the best found so far
        bh_energy, bh_pos = _basin_hopping(best_pos, potential, kind, n_kicks=8, kick_scale=0.4)
        if np.isfinite(bh_energy) and bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Phase 3: More basin hopping with different starting points
    if _time_left() > 15.0:
        # Pick a few more seeds for basin hopping
        extra_seeds = seeds[2:min(len(seeds), 6)]
        for seed_pos in extra_seeds:
            if _time_left() < 8.0:
                break
            bh_energy, bh_pos = _basin_hopping(seed_pos, potential, kind, n_kicks=4, kick_scale=0.6)
            if np.isfinite(bh_energy) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
    
    # Phase 4: Final polish
    if best_pos is not None and _time_left() > 3.0:
        best_pos = _polish(best_pos, potential, kind)
    
    # If we still don't have a valid position, use a fallback
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
