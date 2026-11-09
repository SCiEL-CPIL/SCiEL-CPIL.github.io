"""Rewritten program for Sutton-Chen transition-metal cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Known approximate global-minimum energies from Doye & Wales (1998) for reference
_KNOWN_E = {
    ('12-6', 38): -237.74,
    ('12-6', 75): -502.62,
    ('9-6', 59): -157.93,
    ('9-6', 80): -224.22,
    ('10-8', 55): -164.71,
    ('10-8', 79): -256.20,
}

def _validate_positions(pos):
    """Check if positions are valid (finite, min distance > 0.35)."""
    if not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    # Vectorized check
    for i in range(n):
        dists = np.linalg.norm(pos[i+1:] - pos[i], axis=1)
        if len(dists) > 0 and np.min(dists) < 0.35:
            return False
    return True

def _generate_fcc_cluster(n, lattice=1.0):
    """Generate a cluster from FCC lattice (first n atoms in a compact FCC-like shape)."""
    fcc_points = []
    size = int(np.ceil(n ** (1.0/3.0))) + 2
    coords = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                coords.append([i * lattice, j * lattice, k * lattice])
    fcc_offsets = [
        [0, 0, 0],
        [lattice/2, lattice/2, 0],
        [lattice/2, 0, lattice/2],
        [0, lattice/2, lattice/2],
    ]
    all_points = []
    for base in coords:
        for off in fcc_offsets:
            all_points.append(np.array(base) + np.array(off))
    all_points = np.array(all_points)
    dists = np.linalg.norm(all_points, axis=1)
    # Keep points within a reasonable radius
    max_r = 0.85 * n ** (1.0/3.0) + 1.0
    all_points = all_points[dists <= max_r]
    if len(all_points) >= n:
        d2 = np.linalg.norm(all_points, axis=1)
        order = np.argsort(d2)
        return all_points[order[:n]]
    else:
        return _generate_random_cluster(n, seed=42)

def _generate_hcp_cluster(n, lattice=1.0):
    """Generate a cluster from HCP lattice."""
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    size = int(np.ceil(n ** (1.0/3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    max_r = 0.85 * n ** (1.0/3.0) + 1.0
    points = points[dists <= max_r]
    d2 = np.linalg.norm(points, axis=1)
    order = np.argsort(d2)
    if len(points) >= n:
        return points[order[:n]]
    else:
        return _generate_random_cluster(n, seed=43)

def _generate_random_cluster(n, seed, min_dist=0.8, max_radius=None):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    if max_radius is None:
        max_radius = 0.85 * n ** (1.0 / 3.0)
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_radius, max_radius, 3)
            points.append(p)
    return np.array(points)

def _generate_icosahedral_cluster(n):
    """Generate an icosahedral-like cluster (concentric icosahedra)."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    points = [np.zeros(3)]
    for v in ico_verts:
        points.append(v)
    
    n_verts = len(ico_verts)
    shell2 = []
    for i in range(n_verts):
        for j in range(i+1, n_verts):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 0.9 < d < 1.1:
                mid = (ico_verts[i] + ico_verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.35
                shell2.append(mid)
    for p in shell2:
        points.append(p)
    
    shell3 = []
    for i in range(n_verts):
        for j in range(i+1, n_verts):
            for k in range(j+1, n_verts):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if all(0.9 < d < 1.1 for d in [d1, d2, d3]):
                    face_center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                    face_center = face_center / np.linalg.norm(face_center) * 1.7
                    shell3.append(face_center)
    for p in shell3:
        points.append(p)
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        rng = np.random.default_rng(44)
        while len(points) < n:
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * 2.0
            points.append(p)
    return np.array(points[:n])

def construct(instance):
    """
    Multi-start optimization: generate many diverse starting configurations,
    relax each with L-BFGS-B, keep the best energy found within time budget.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # 320s to leave 10s safety margin
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    # Generate diverse starting configurations
    candidates = []
    
    # 1. FCC-based
    candidates.append(_generate_fcc_cluster(n, lattice=0.72))
    # 2. HCP-based
    candidates.append(_generate_hcp_cluster(n, lattice=0.72))
    # 3. Icosahedral
    candidates.append(_generate_icosahedral_cluster(n))
    
    # 4. Random clusters with varying seeds and min distances
    min_dists = [0.7, 0.8, 0.9, 1.0]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    best_energy = np.inf
    best_positions = None
    
    for seed in seeds:
        for md in min_dists:
            if time.time() > deadline:
                break
            pos = _generate_random_cluster(n, seed=seed, min_dist=md)
            if _validate_positions(pos):
                candidates.append(pos)
    
    # Sort candidates: put structured ones first (more likely to be good)
    for pos in candidates:
        if time.time() > deadline:
            break
        try:
            if not _validate_positions(pos):
                continue
            energy, relaxed = relax(pos, potential, kind)
            if np.isfinite(energy) and _validate_positions(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            continue
    
    if best_positions is None:
        pos = _generate_random_cluster(n, seed=99, min_dist=0.8)
        try:
            energy, best_positions = relax(pos, potential, kind)
        except:
            best_positions = pos
    
    return best_positions
# EVOLVE-BLOCK-END
